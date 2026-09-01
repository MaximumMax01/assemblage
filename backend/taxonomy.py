import os
<<<<<<< HEAD
import sys
import re
import torch
from typing import Any, Dict, List

# Ensure backend directory is discoverable for imports
# This line allows Python to find 'purformat' and other modules correctly
=======
import re
import sys
from typing import Any, Dict, List, NamedTuple

import torch

>>>>>>> b3ed0a9 (second commit, for school)
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

<<<<<<< HEAD
# Dynamically import OpenCLIP (it's already loaded by validator.py)
# We need its tokenizer and potentially model if not passed
try:
    import open_clip
except ImportError:
    # This should not happen if requirements.txt is installed, but for robustness
    raise ImportError("OpenCLIP not found. Ensure 'open-clip-torch' is installed.")


# Archetype descriptions and their associated search templates
# These descriptions are fed into OpenCLIP to create vector "fingerprints"
# The templates are designed to guide searches for specific types of 3D reference
ARCHETYPE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "architectural_interior": {
        "description": "detailed architectural interior finishing elements, wall trim, baseboards, crown molding, door frames, window sills, paneling, joinery, room structure, abandoned interior, liminal space.",
        "templates": [
            "{prompt} architectural interior detail 3/4 perspective neutral lighting",
            "{prompt} baseboard crown molding wall trim cross section profile schematic",
            "{prompt} corner joint miter seam construction detail blueprint",
            "{prompt} textured surface material macro photogrammetry scan"
        ]
    },
    "vehicle_mechanical": {
        "description": "industrial machinery, vehicles, engines, robots, hydraulics, gearboxes, mechanisms, cyberpunk tech, sci-fi mechanical parts, metal structure, worn components, exploded view.",
        "templates": [
            "{prompt} orthographic technical blueprint front side top",
            "{prompt} mechanical cutaway schematic manual exploded view",
            "{prompt} engine suspension chassis linkage component detail closeup",
            "{prompt} rusted metal worn paint texture macro photogrammetry"
        ]
    },
    "organic_natural": {
        "description": "trees, forests, rocks, terrain, foliage, bark, leaves, plants, natural environments, wilderness, moss, overgrown surfaces, organic shapes, riverbed, redwood.",
        "templates": [
            "{prompt} botanical specimen isolated natural lighting",
            "{prompt} surface texture macro 4k photogrammetry scan PBR",
            "{prompt} forest landscape environment structure perspective",
            "{prompt} erosion crack weathering detail close up"
        ]
    },
    "prop_asset": {
        "description": "props, weapons, swords, knives, furniture, tools, electronic devices, historical artifacts, studio reference, detailed models.",
        "templates": [
            "{prompt} full length historical replica white background studio",
            "{prompt} orthographic blueprint profile front side elevation",
            "{prompt} hilt grip handle crossguard construction detail closeup",
            "{prompt} material texture surface wear macro photogrammetry"
        ]
    },
    # Generic fallback if no specific archetype matches well
    "generic_detail": {
        "description": "general detailed reference image, object study, technical illustration, construction photo, material research, aesthetic moodboard.",
        "templates": [
            "{prompt} high quality detailed reference photo",
            "{prompt} technical drawing blueprint schematic",
            "{prompt} close-up construction detail assembly",
            "{prompt} material texture macro wear"
        ]
    }
}

class TaxonomyEngine:
    def __init__(self, clip_model: Any, clip_tokenizer: Any, device: str) -> None:
        """
        Initializes the TaxonomyEngine with the pre-loaded OpenCLIP model.
        Args:
            clip_model: The OpenCLIP model instance (already on device).
            clip_tokenizer: The OpenCLIP tokenizer instance.
            device: The torch device ('mps', 'cuda', or 'cpu').
        """
=======
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
>>>>>>> b3ed0a9 (second commit, for school)
        self.model = clip_model
        self.tokenizer = clip_tokenizer
        self.device = device
        self.archetype_vectors = self._preload_archetype_vectors()
<<<<<<< HEAD
        print(f"[TaxonomyEngine] Loaded {len(self.archetype_vectors)} semantic archetypes.")


    def _preload_archetype_vectors(self) -> Dict[str, torch.Tensor]:
        """
        Precomputes and stores vector embeddings for each archetype description.
        This is done once at startup for efficiency.
        """
=======
        print(f"[Taxonomy] Loaded {len(self.archetype_vectors)} semantic archetypes.")

    def _preload_archetype_vectors(self) -> Dict[str, torch.Tensor]:
>>>>>>> b3ed0a9 (second commit, for school)
        vectors: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, data in ARCHETYPE_DEFINITIONS.items():
                tokens = self.tokenizer(data["description"]).to(self.device)
<<<<<<< HEAD
                # Encode the text description into a vector (its "fingerprint")
                features: torch.Tensor = self.model.encode_text(tokens)
                # Normalize the vector so its length is 1 (important for cosine similarity)
                vectors[name] = features / features.norm(dim=-1, keepdim=True)
        return vectors

    async def generate_reference_queries(self, prompt: str) -> List[str]:
        """
        Analyzes the user's prompt using OpenCLIP to find the best matching
        3D archetype and generates specialized search queries.
        """
        clean_prompt = re.sub(r'[^\w\s]', '', prompt).strip()

        # 1. Embed the user's prompt into a vector
=======
                features: torch.Tensor = self.model.encode_text(tokens)
                vectors[name] = features / features.norm(dim=-1, keepdim=True)
        return vectors

    def classify(self, prompt: str) -> str:
        """Returns the best-matching archetype name for a prompt."""
        clean_prompt = re.sub(r"[^\w\s]", "", prompt).strip()

>>>>>>> b3ed0a9 (second commit, for school)
        with torch.no_grad():
            prompt_tokens = self.tokenizer(clean_prompt).to(self.device)
            prompt_feat: torch.Tensor = self.model.encode_text(prompt_tokens)
            prompt_feat = prompt_feat / prompt_feat.norm(dim=-1, keepdim=True)

<<<<<<< HEAD
        # 2. Find the best matching archetype by comparing vector fingerprints
        best_archetype = "generic_detail"
        max_similarity = -1.0

        for name, archetype_vector in self.archetype_vectors.items():
            # Calculate cosine similarity (dot product of normalized vectors)
=======
        best_archetype = "generic_detail"
        max_similarity = -1.0
        for name, archetype_vector in self.archetype_vectors.items():
>>>>>>> b3ed0a9 (second commit, for school)
            similarity = float((prompt_feat @ archetype_vector.T).item())
            if similarity > max_similarity:
                max_similarity = similarity
                best_archetype = name
<<<<<<< HEAD
        
        # Add a confidence threshold: if the best match is too weak, fallback to generic
        if max_similarity < 0.25:  # This threshold can be fine-tuned
             best_archetype = "generic_detail"
        
        print(f"[TaxonomyEngine] Prompt '{prompt}' matched to archetype: '{best_archetype}' (Similarity: {max_similarity:.2f})")

        # 3. Use the templates from the best matching archetype
        selected_templates = ARCHETYPE_DEFINITIONS[best_archetype]["templates"]
        
        return [template.format(prompt=clean_prompt) for template in selected_templates]
=======

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
>>>>>>> b3ed0a9 (second commit, for school)
