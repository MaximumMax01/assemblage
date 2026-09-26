"""
Slot definitions for Assemblage.

A "slot" is one of the four orthogonal kinds of reference a modeller needs for a
subject. The whole point of the tool is that a board contains one of each rather
than twelve variations of the same hero shot, so the slot tag has to survive the
entire pipeline: query generation -> search -> download -> filtering -> scoring
-> layout.

Each slot carries its own scoring profile because the filters that make sense for
a photograph actively destroy a line drawing. A blueprint is mostly white paper,
has low global edge variance, and looks a lot like an e-commerce cutout to CLIP.
Filtering it with the HERO profile removes it every time.
"""

from typing import Dict, List, Optional, TypedDict


class SlotProfile(TypedDict):
    label: str
    description: str
    positives: List[str]
    negatives: List[str]
    white_gate: Optional[float]
    min_laplacian: float
    max_aspect: float
    min_dim: int
    min_score: float


# Negatives that apply to every slot. Kept separate from the white-background
# negative, which is only correct for photographic slots.
_UNIVERSAL_NEGATIVES = [
    "blurry low resolution thumbnail with jpeg artifacts",
    "text watermark meme user interface graphic",
    "video game inventory skin card market price listing",
    "stock photography watermark overlay grid",
]

HERO = "hero"
ORTHO = "ortho"
DETAIL = "detail"
MATERIAL = "material"

# Floor-plan mode. "Floor plan" is a view type, not an object, so the object
# slots do not apply to it: a floor plan has no hero shot and no joinery detail.
# These four split plans the way a modeller actually uses them.
PLAN_SIMPLE = "plan_simple"
PLAN_DIMS = "plan_dims"
PLAN_FURNISHED = "plan_furnished"
PLAN_CUTAWAY = "plan_cutaway"

# Kept for backwards compatibility: the default (object) mode's slot order.
SLOT_ORDER = [HERO, ORTHO, DETAIL, MATERIAL]

SLOT_PROFILES: Dict[str, SlotProfile] = {
    HERO: {
        "label": "Form & silhouette",
        "description": "the whole object, three-quarter view",
        "positives": [
            "sharp photograph of {prompt}",
            "three quarter perspective view of {prompt}",
            "full object reference photograph of {prompt} in neutral lighting",
        ],
        # Hero shots are photographs, so an e-commerce cutout is a real failure
        # mode here and worth penalising.
        "negatives": _UNIVERSAL_NEGATIVES + [
            "e-commerce product listing on pure white background",
        ],
        "white_gate": 0.70,
        "min_laplacian": 90.0,
        "max_aspect": 3.2,
        "min_dim": 600,
        "min_score": 0.05,
    },
    ORTHO: {
        "label": "Orthographic & technical",
        "description": "blueprints, elevations and cross-sections",
        "positives": [
            "orthographic technical drawing of {prompt}",
            "blueprint schematic diagram of {prompt}",
            "cross section elevation drawing of {prompt}",
            "measured plan and side view of {prompt}",
        ],
        # No white-background negative: line drawings live on white paper.
        "negatives": _UNIVERSAL_NEGATIVES,
        # White gate disabled entirely. This is the single most important line
        # in this file -- a 0.70 gate deletes almost every blueprint.
        "white_gate": None,
        # Line art is sparse: strong edges but few of them, so global Laplacian
        # variance runs much lower than a photograph's.
        "min_laplacian": 25.0,
        # Elevation sheets and multi-view plates are often very wide.
        "max_aspect": 5.0,
        "min_dim": 500,
        "min_score": 0.03,
    },
    DETAIL: {
        "label": "Joinery & construction",
        "description": "corners, seams, how it goes together",
        "positives": [
            "close up photograph of the construction detail of {prompt}",
            "joint seam and assembly detail of {prompt}",
            "mechanical linkage closeup of {prompt}",
        ],
        "negatives": _UNIVERSAL_NEGATIVES + [
            "e-commerce product listing on pure white background",
        ],
        "white_gate": 0.75,
        "min_laplacian": 90.0,
        "max_aspect": 3.2,
        "min_dim": 600,
        "min_score": 0.05,
    },
    MATERIAL: {
        "label": "Material & surface",
        "description": "texture, wear and roughness up close",
        "positives": [
            "macro surface texture of {prompt}",
            "photogrammetry material scan of {prompt}",
            "close up of surface wear and roughness on {prompt}",
        ],
        "negatives": _UNIVERSAL_NEGATIVES + [
            "e-commerce product listing on pure white background",
        ],
        "white_gate": 0.75,
        # Texture plates are high-frequency by definition, so this gate can be
        # stricter than the photographic default without losing anything.
        "min_laplacian": 120.0,
        "max_aspect": 2.5,
        "min_dim": 600,
        "min_score": 0.05,
    },

    # ---- floor-plan mode -------------------------------------------------
    # Every plan slot has the white gate disabled for the same reason ORTHO
    # does: plans are drawings, and drawings are mostly white paper.
    PLAN_SIMPLE: {
        "label": "Simple plans",
        "description": "clean line plans that are easy to block out",
        "positives": [
            "simple black and white floor plan of {prompt}",
            "clean architectural line drawing floor plan of {prompt}",
            "minimal schematic room layout of {prompt}",
        ],
        # Pushing against colour and 3D is what separates this slot from the
        # furnished and cutaway slots, which otherwise match the same images.
        "negatives": _UNIVERSAL_NEGATIVES + [
            "photorealistic 3d rendered interior",
            "colorful furnished real estate floor plan",
        ],
        "white_gate": None,
        "min_laplacian": 20.0,
        "max_aspect": 4.0,
        "min_dim": 500,
        "min_score": 0.03,
    },
    PLAN_DIMS: {
        "label": "Dimensioned",
        "description": "measured plans, so rooms come out the right size",
        "positives": [
            "floor plan of {prompt} with dimension lines and measurements",
            "architectural construction drawing plan of {prompt} with dimensions",
            "measured drawing floor plan of {prompt}",
        ],
        "negatives": _UNIVERSAL_NEGATIVES + [
            "photorealistic 3d rendered interior",
        ],
        "white_gate": None,
        "min_laplacian": 20.0,
        "max_aspect": 4.0,
        "min_dim": 500,
        "min_score": 0.03,
    },
    PLAN_FURNISHED: {
        "label": "Furnished",
        "description": "furniture and fixtures placed in each room",
        "positives": [
            "furnished floor plan of {prompt} with furniture layout",
            "colored floor plan of {prompt} showing furniture and fixtures",
            "interior layout plan of {prompt} with desks and seating",
        ],
        "negatives": _UNIVERSAL_NEGATIVES + [
            "photorealistic 3d rendered interior",
        ],
        "white_gate": None,
        "min_laplacian": 30.0,
        "max_aspect": 4.0,
        "min_dim": 500,
        "min_score": 0.03,
    },
    PLAN_CUTAWAY: {
        "label": "3D cutaway",
        "description": "wall heights and room volumes, seen from above",
        "positives": [
            "3d cutaway floor plan of {prompt}",
            "isometric axonometric floor plan render of {prompt}",
            "dollhouse view 3d floor plan of {prompt}",
        ],
        "negatives": _UNIVERSAL_NEGATIVES + [
            "flat 2d black and white line drawing",
        ],
        "white_gate": None,
        "min_laplacian": 40.0,
        "max_aspect": 3.2,
        "min_dim": 500,
        "min_score": 0.03,
    },
}


OBJECT_MODE = "object"
PLAN_MODE = "plan"

# A mode bundles the slots a board is split into with the anchor used to judge
# subject relevance. The subject anchor differs by mode: for an object the
# question is "is this that thing", for a plan it is "is this a floor plan of
# that kind of building". A photograph of a hotel lobby should fail the second.
MODES: Dict[str, dict] = {
    OBJECT_MODE: {
        "label": "Object reference",
        "slots": [HERO, ORTHO, DETAIL, MATERIAL],
        "subject_templates": ["{prompt}", "a photograph of {prompt}"],
    },
    PLAN_MODE: {
        "label": "Floor plans",
        "slots": [PLAN_SIMPLE, PLAN_DIMS, PLAN_FURNISHED, PLAN_CUTAWAY],
        "subject_templates": ["floor plan of {prompt}", "{prompt} floor plan"],
    },
}


def profile_for(slot: str) -> SlotProfile:
    """
    Returns the scoring profile for a slot.

    Raises on an unknown slot rather than falling back to a default. Slot ids now
    arrive from the browser when a user edits their searches, so a silent
    fallback would let a typo or a stale page quietly score everything with the
    wrong filters.
    """
    try:
        return SLOT_PROFILES[slot]
    except KeyError:
        raise ValueError(f"Unknown slot: {slot!r}") from None


def mode_of(slot: str) -> str:
    """Returns the mode a slot belongs to."""
    for name, mode in MODES.items():
        if slot in mode["slots"]:
            return name
    raise ValueError(f"Slot {slot!r} belongs to no mode")


def slots_for(mode: str) -> List[str]:
    return list(MODES[mode]["slots"])
