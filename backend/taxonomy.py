import os
import re
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import open_clip
except ImportError:
    raise ImportError("OpenCLIP not found. Ensure 'open-clip-torch' is installed.")

from slots import (
    DETAIL, HERO, MATERIAL, MODES, OBJECT_MODE, ORTHO, PLAN_CUTAWAY, PLAN_DIMS,
    PLAN_FURNISHED, PLAN_MODE, PLAN_SIMPLE, mode_of, slots_for,
)


class SlotQuery(NamedTuple):
    """A search query tagged with the reference slot it is meant to fill."""
    slot: str
    query: str


class QueryPlan(NamedTuple):
    """
    Everything decided about a prompt before any searching happens.

    Returned to the browser so the user can see, and edit, exactly what will be
    searched. Most bad boards trace back to a prompt that did not say what the
    user meant; showing the searches up front is how they find that out before
    spending a run on it rather than after.
    """
    mode: str
    archetype: str
    subject: str
    queries: List[SlotQuery]
    hints: List[dict]
    # What the interface calls this kind of subject, e.g. "Light fixture".
    kind_label: str = ""
    # Archetype-specific overrides for scoring. Empty means "use the mode's".
    subject_templates: Tuple[str, ...] = ()
    extra_negatives: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

# Phrases that turn a prompt into a request for a view type rather than an
# object. Longest first, so "floor plans" is matched before "floor plan".
PLAN_TRIGGERS = sorted([
    "floor plans", "floor plan", "floorplans", "floorplan", "floor layout",
    "room layout", "layout plan", "site plan", "building plan", "plan view",
], key=len, reverse=True)

PLAN_TEMPLATES: Dict[str, str] = {
    PLAN_SIMPLE: "{prompt} simple floor plan",
    PLAN_DIMS: "{prompt} floor plan dimensions",
    PLAN_FURNISHED: "{prompt} furnished floor plan",
    PLAN_CUTAWAY: "{prompt} 3d floor plan",
}

_LEADING_FILLER = re.compile(r"^(?:(?:of|for|a|an|the)\s+)+")


def clean_prompt(prompt: str) -> str:
    """Strips punctuation and collapses whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", prompt)).strip()


def detect_mode(prompt: str) -> Tuple[str, str]:
    """
    Returns (mode, subject).

    "Floor plan hotel" is a request for a view type about a subject, so the view
    type is removed and becomes the mode, leaving "hotel" as the thing searched
    for. Everything else is an object prompt and the whole cleaned prompt is
    the subject.
    """
    clean = clean_prompt(prompt).lower()
    for trigger in PLAN_TRIGGERS:
        pattern = r"\b" + re.escape(trigger) + r"\b"
        if re.search(pattern, clean):
            rest = re.sub(r"\s+", " ", re.sub(pattern, " ", clean)).strip()
            rest = _LEADING_FILLER.sub("", rest).strip()
            return PLAN_MODE, rest
    return OBJECT_MODE, clean


# ---------------------------------------------------------------------------
# Prompt hints
# ---------------------------------------------------------------------------

# Words that describe how a place feels rather than what is in it. Search
# engines answer them with game screenshots, mood boards and AI art, none of
# which is usable modelling reference.
MOOD_WORDS = {
    "liminal", "creepy", "eerie", "spooky", "aesthetic", "vibe", "vibes",
    "cinematic", "moody", "dreamcore", "weirdcore", "atmospheric", "cozy",
    "dreamy", "surreal", "backrooms", "nostalgic", "unsettling", "abandoned",
}

# Concrete replacements for the mood words people most often reach for. These
# are the vocabulary gap in practice: "tenant improvement" is the architectural
# term for partition walls built out inside open commercial space, which is
# exactly what the backrooms look like, and nobody searches for it unless they
# already know the phrase.
MOOD_SUGGESTIONS: Dict[str, Dict[str, List[str]]] = {
    "backrooms": {
        OBJECT_MODE: ["commercial carpet tile", "drop ceiling grid",
                      "fluorescent troffer light", "office partition wall"],
        PLAN_MODE: ["tenant improvement floor plan", "office suite floor plan",
                    "small medical office floor plan", "strip mall unit floor plan"],
    },
    "liminal": {
        OBJECT_MODE: ["hotel corridor carpet", "school locker bank",
                      "drop ceiling grid", "fire exit door"],
        PLAN_MODE: ["hotel corridor floor plan", "school floor plan",
                    "motel floor plan", "office suite floor plan"],
    },
    "abandoned": {
        OBJECT_MODE: ["water damaged drywall", "peeling paint texture",
                      "collapsed ceiling tile", "rusted door hinge"],
        PLAN_MODE: ["warehouse floor plan", "factory floor plan",
                    "office building floor plan", "school floor plan"],
    },
}

# Head nouns that name a whole space. English noun phrases put the head last,
# so "office ceiling tiles" is about tiles (fine) while "modern office" is about
# an office (a scene). Checking only the last word avoids flagging the first.
SCENE_NOUNS_ENCLOSED = {
    "house", "home", "building", "room", "kitchen", "bathroom", "bedroom",
    "interior", "lobby", "office", "hallway", "corridor", "apartment",
    "warehouse", "mall", "school", "hospital", "hotel", "garage", "basement",
    "level", "motel", "classroom", "cubicle",
}
# Outdoor and exterior scenes have no floor plan, so suggesting one would be
# nonsense ("house front floor plan"). They still get the one-part-at-a-time
# advice, just without that suggestion.
SCENE_NOUNS_OPEN = {
    "street", "city", "town", "exterior", "facade", "front", "scene",
    "environment", "landscape", "village", "backyard", "yard", "park",
}
SCENE_NOUNS = SCENE_NOUNS_ENCLOSED | SCENE_NOUNS_OPEN

MAX_HINTS = 2


def prompt_hints(prompt: str, mode: str, subject: str) -> List[dict]:
    """
    Flags prompts that are likely to produce a board the user did not want.

    Each hint is {"kind", "text", "suggestions"}; suggestions are complete
    prompts the interface can offer as one-click replacements. Deterministic
    and local on purpose: no model call, so it can run on every keystroke.
    """
    words = clean_prompt(prompt).lower().split()
    hints: List[dict] = []

    mood = [w for w in words if w in MOOD_WORDS]
    if mood:
        suggestions: List[str] = []
        for w in mood:
            suggestions += MOOD_SUGGESTIONS.get(w, {}).get(mode, [])
        quoted = " and ".join(f"\u201c{w}\u201d" for w in mood)
        verb = "describes" if len(mood) == 1 else "describe"
        hints.append({
            "kind": "mood",
            "text": (f"{quoted} {verb} a feeling rather than a thing. Searches for "
                     "mood words return game screenshots and AI art, not reference. "
                     "Name what is physically there instead: a material, a "
                     "fixture, a part."),
            "suggestions": list(dict.fromkeys(suggestions))[:4],
        })

    if mode == PLAN_MODE and not subject:
        hints.append({
            "kind": "vague",
            "text": ("Say what kind of building. Plans for a specific use are "
                     "far more consistent than plans for anything at all."),
            "suggestions": ["office suite floor plan", "school floor plan",
                            "warehouse floor plan", "motel floor plan"],
        })
    elif mode == OBJECT_MODE and words and words[-1] in SCENE_NOUNS:
        enclosed = words[-1] in SCENE_NOUNS_ENCLOSED
        text = ("This sounds like a whole space, not one object. Boards work best "
                "on a single part, so search the pieces one at a time.")
        if enclosed:
            text += " For the room layout itself, ask for a floor plan."
        hints.append({
            "kind": "scene",
            "text": text,
            "suggestions": [f"{subject} floor plan"] if (enclosed and subject) else [],
        })
    elif mode == OBJECT_MODE and len(words) == 1 and not mood:
        hints.append({
            "kind": "vague",
            "text": ("One word is usually too broad. Add a material, a style or a "
                     "part: \u201coak dining chair\u201d finds better reference "
                     "than \u201cchair\u201d."),
            "suggestions": [],
        })

    return hints[:MAX_HINTS]


# ---------------------------------------------------------------------------
# Edited searches from the browser
# ---------------------------------------------------------------------------

MAX_QUERY_CHARS = 200


def resolve_overrides(overrides: Sequence[Tuple[str, str]]) -> Tuple[str, List[SlotQuery]]:
    """
    Validates searches the user edited in the browser.

    Returns (mode, queries). A blank query means the user switched that slot
    off, and is dropped. Raises ValueError on anything malformed: an unknown
    slot, slots from two different modes, a duplicate, an over-long query, or
    every slot switched off.
    """
    queries: List[SlotQuery] = []
    seen: set = set()
    modes: set = set()
    for slot, query in overrides:
        modes.add(mode_of(slot))                 # raises on an unknown slot
        if slot in seen:
            raise ValueError(f"Slot {slot!r} appears twice")
        seen.add(slot)
        q = re.sub(r"\s+", " ", query or "").strip()
        if len(q) > MAX_QUERY_CHARS:
            raise ValueError(f"Search for {slot!r} is longer than {MAX_QUERY_CHARS} characters")
        if q:
            queries.append(SlotQuery(slot=slot, query=q))
    if len(modes) > 1:
        raise ValueError("Searches mix floor-plan and object slots")
    if not queries:
        raise ValueError("Every search is blank, so there is nothing to look for")
    return modes.pop(), queries


# Each archetype supplies exactly one template per slot. Keying by slot rather
# than relying on list order means an archetype can no longer accidentally ship
# two orthographic queries and no hero shot, which is what vehicle_mechanical
# was doing before.
ARCHETYPE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "architectural_interior": {
        "label": "Architectural detail",
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
        "label": "Mechanical",
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
        "label": "Natural",
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
        "label": "Prop",
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
    # Light fixtures need their own archetype because "light" names two things:
    # the object, and the glow it makes. Search engines and CLIP both lean hard
    # towards the glow -- lit stages, beams, people in coloured light -- because
    # that is what most photos captioned "stage light" actually show. Every
    # template here says "fixture" or names a physical part, the subject anchor
    # asks for the fixture rather than the phrase, and the extra negatives push
    # directly against the effect.
    "lighting_fixture": {
        "label": "Light fixture",
        "description": (
            "light fixtures and lamps as physical objects: stage lights, spotlights, "
            "PAR cans, fresnels, floodlights, chandeliers, sconces, pendant lamps, "
            "street lamps, lanterns, housings with lenses and mounting brackets."
        ),
        "templates": {
            HERO: "{prompt} fixture product photo",
            ORTHO: "{prompt} fixture dimension drawing",
            DETAIL: "{prompt} mounting bracket clamp",
            MATERIAL: "{prompt} housing metal finish",
        },
        "subject_templates": ["{prompt} fixture", "a product photograph of a {prompt} fixture"],
        "negatives": [
            "concert stage lit with colored light beams and haze",
            "glowing light beam effect on a black background",
            "person lit by colored stage lighting",
        ],
    },
    "generic_detail": {
        "label": "Object reference",
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


# Head nouns that name a light-emitting object. Checked on the LAST word only, the
# same head-noun rule the scene hints use: "stage light" is a light, but "light
# oak table" is a table and "light switch" is a switch. This is a word rule, not
# a model call, so it is predictable and testable -- CLIP classification is only
# the fallback for everything these lists do not cover.
LIGHTING_NOUNS = {
    "light", "lights", "lamp", "lamps", "spotlight", "spotlights", "floodlight",
    "floodlights", "chandelier", "sconce", "sconces", "lantern", "lanterns",
    "luminaire", "troffer", "fixture", "fixtures", "streetlight", "headlight",
}


def route_archetype(subject: str) -> Optional[str]:
    """Returns an archetype decided by word rules, or None to fall back to CLIP."""
    words = subject.lower().split()
    if words and words[-1] in LIGHTING_NOUNS:
        return "lighting_fixture"
    return None


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

    def plan(self, prompt: str) -> QueryPlan:
        """
        Decides mode, subject and the searches for a prompt, without running them.

        Cheap: at most one text-encoder pass, for the archetype. Called on every
        pause in typing so the searches update as the user writes.
        """
        mode, subject = detect_mode(prompt)
        if mode == PLAN_MODE:
            archetype = "floor_plan"
            templates = PLAN_TEMPLATES
            fill = subject or "building"
        else:
            archetype = route_archetype(subject) or (
                self.classify(subject) if subject else "generic_detail")
            templates = ARCHETYPE_DEFINITIONS[archetype]["templates"]
            fill = subject

        queries = [
            SlotQuery(slot=slot, query=clean_prompt(templates[slot].format(prompt=fill)))
            for slot in slots_for(mode)
            if slot in templates
        ]
        arch = ARCHETYPE_DEFINITIONS.get(archetype, {})
        return QueryPlan(
            mode=mode,
            archetype=archetype,
            subject=subject or ("building" if mode == PLAN_MODE else ""),
            queries=queries,
            hints=prompt_hints(prompt, mode, subject),
            kind_label=MODES[mode]["label"] if mode == PLAN_MODE else arch.get("label", ""),
            subject_templates=tuple(arch.get("subject_templates", ())),
            extra_negatives=tuple(arch.get("negatives", ())),
        )

    async def generate_reference_queries(self, prompt: str) -> List[SlotQuery]:
        """Kept for callers that only need the searches."""
        return self.plan(prompt).queries
