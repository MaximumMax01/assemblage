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

SLOT_ORDER = [HERO, ORTHO, DETAIL, MATERIAL]

SLOT_PROFILES: Dict[str, SlotProfile] = {
    HERO: {
        "label": "Form & silhouette",
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
}


def profile_for(slot: str) -> SlotProfile:
    """Returns the scoring profile for a slot, falling back to HERO."""
    return SLOT_PROFILES.get(slot, SLOT_PROFILES[HERO])
