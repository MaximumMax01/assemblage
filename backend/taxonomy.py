import os
import re
import sys
from typing import Any, Dict, List, NamedTuple

import torch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import open_clip
except ImportError:
    raise ImportError("OpenCLIP not found. Ensure 'open-clip-torch' is installed.")

from slots import HERO, ORTHO, DETAIL, MATERIAL, SLOT_ORDER


class SlotQuery(NamedTuple):
    """A search query tagged with the reference slot it is meant to fill."""
    slot: str
    query: str


# Each archetype supplies exactly one template per slot. Keying by slot rather
# than relying on list order means an archetype can no longer accidentally ship
# two orthographic queries and no hero shot, which is what vehicle_mechanical
# was doing before.
ARCHETYPE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "architectural_interior": {
        "description": (
            "detailed architectural interior finishing elements, wall trim, baseboards, "
            "crown molding, door frames, window sills, paneling, joinery, room structure, "
            "abandoned interior, liminal space."
        ),
        "templates": {
            HERO: "{prompt} interior photograph",
            ORTHO: "{prompt} section detail drawing",
            DETAIL: "{prompt} installation detail closeup",
            MATERIAL: "{prompt} surface texture macro",
        },
    },
    "vehicle_mechanical": {
        "description": (
            "industrial machinery, vehicles, engines, robots, hydraulics, gearboxes, "
            "mechanisms, cyberpunk tech, sci-fi mechanical parts, metal structure, "
            "worn components, exploded view."
        ),
        "templates": {
            # Previously missing entirely: every vehicle prompt produced two
            # near-identical blueprint queries and no full-object photograph.
            HERO: "{prompt} photograph full view",
            ORTHO: "{prompt} blueprint orthographic drawing",
            DETAIL: "{prompt} component closeup detail",
            MATERIAL: "{prompt} worn metal texture macro",
        },
    },
    "organic_natural": {
        "description": (
            "trees, forests, rocks, terrain, foliage, bark, leaves, plants, natural "
            "environments, wilderness, moss, overgrown surfaces, organic shapes, "
            "riverbed, redwood."
        ),
        "templates": {
            HERO: "{prompt} photograph",
            # Organic subjects rarely have blueprints, but scientific illustration
            # plates fill the same structural role.
            ORTHO: "{prompt} scientific illustration diagram",
            DETAIL: "{prompt} structure closeup detail",
            MATERIAL: "{prompt} surface texture macro scan",
        },
    },
    "prop_asset": {
        "description": (
            "props, weapons, swords, knives, furniture, tools, electronic devices, "
            "historical artifacts, studio reference, detailed models."
        ),
        "templates": {
            HERO: "{prompt} full photograph",
            ORTHO: "{prompt} blueprint profile drawing",
            DETAIL: "{prompt} construction detail closeup",
            MATERIAL: "{prompt} material texture macro",
        },
    },
    "generic_detail": {
        "description": (
            "general detailed reference image, object study, technical illustration, "
            "construction photo, material research, aesthetic moodboard."
        ),
        "templates": {
            HERO: "{prompt} photograph",
            ORTHO: "{prompt} technical drawing blueprint",
            DETAIL: "{prompt} construction detail closeup",
            MATERIAL: "{prompt} material texture macro",
        },
    },
}


class TaxonomyEngine:
    def __init__(self, clip_model: Any, clip_tokenizer: Any, device: str) -> None:
        self.model = clip_model
        self.tokenizer = clip_tokenizer
        self.device = device
        self.archetype_vectors = self._preload_archetype_vectors()
        print(f"[Taxonomy] Loaded {len(self.archetype_vectors)} semantic archetypes.")

    def _preload_archetype_vectors(self) -> Dict[str, torch.Tensor]:
        vectors: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, data in ARCHETYPE_DEFINITIONS.items():
                tokens = self.tokenizer(data["description"]).to(self.device)
                features: torch.Tensor = self.model.encode_text(tokens)
                vectors[name] = features / features.norm(dim=-1, keepdim=True)
        return vectors

    def classify(self, prompt: str) -> str:
        """Returns the best-matching archetype name for a prompt."""
        clean_prompt = re.sub(r"[^\w\s]", "", prompt).strip()

        with torch.no_grad():
            prompt_tokens = self.tokenizer(clean_prompt).to(self.device)
            prompt_feat: torch.Tensor = self.model.encode_text(prompt_tokens)
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)

        best_archetype = "generic_detail"
        max_similarity = -1.0
        for name, archetype_vector in self.archetype_vectors.items():
            similarity = float((prompt_feat @ archetype_vector.T).item())
            if similarity > max_similarity:
                max_similarity = similarity
                best_archetype = name

        if max_similarity < 0.25:
            best_archetype = "generic_detail"

        print(f"[Taxonomy] '{prompt}' -> {best_archetype} (sim {max_similarity:.3f})")
        return best_archetype

    async def generate_reference_queries(self, prompt: str) -> List[SlotQuery]:
        """
        Produces one slot-tagged query per reference slot.

        The slot tag is the important part of the return value. Downstream it
        decides which scoring profile an image is judged against and which quota
        bucket it competes in, so it must not be discarded.
        """
        clean_prompt = re.sub(r"[^\w\s]", "", prompt).strip()
        templates = ARCHETYPE_DEFINITIONS[self.classify(prompt)]["templates"]

        return [
            SlotQuery(slot=slot, query=templates[slot].format(prompt=clean_prompt))
            for slot in SLOT_ORDER
            if slot in templates
        ]