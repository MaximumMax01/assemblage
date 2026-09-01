"""
Tests for the slot-aware filtering and quota selection.

torch and open_clip are stubbed so this runs without a 2GB install. The parts
under test here are the ones that decide what ends up on the board: the
per-slot CV gates, the duplicate pass, and the quota allocation.
"""

import io
import os
import sys
import types
from typing import List

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))


# --------------------------------------------------------------------- #
# Stub torch / open_clip before importing validator
# --------------------------------------------------------------------- #

def _install_stubs() -> None:
    torch = types.ModuleType("torch")

    class _Backends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return False

    torch.backends = _Backends()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = lambda: types.SimpleNamespace(
        __enter__=lambda s: None, __exit__=lambda s, *a: False
    )
    torch.Tensor = object
    torch.stack = lambda *a, **k: None
    torch.cat = lambda *a, **k: None
    sys.modules["torch"] = torch

    open_clip = types.ModuleType("open_clip")
    open_clip.create_model_and_transforms = lambda *a, **k: (None, None, None)
    open_clip.get_tokenizer = lambda *a, **k: None
    sys.modules["open_clip"] = open_clip


_install_stubs()

from slots import DETAIL, HERO, MATERIAL, ORTHO, profile_for  # noqa: E402
import validator as V  # noqa: E402


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #

def blueprint(w: int = 1200, h: int = 800) -> bytes:
    """A synthetic line drawing: dark lines on white paper, like a real plate."""
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for i in range(6):
        d.rectangle([60 + i * 60, 60 + i * 40, w - 60 - i * 60, h - 60 - i * 40],
                    outline="black", width=3)
    for i in range(0, w, 40):
        d.line([(i, 0), (i, h)], fill=(210, 210, 210), width=1)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def photo(w: int = 1000, h: int = 1000, seed: int = 0) -> bytes:
    """
    A synthetic photograph: a structured scene plus fine grain.

    Pure random noise is a bad stand-in for a photo when testing perceptual
    hashing -- it has no low-frequency structure for a difference hash to latch
    onto, so JPEG recompression alone shifts the hash by ~6 bits. Real images
    have large-scale structure; this fixture reproduces that.
    """
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=(y % 200, 80 + y // 8, 200 - y // 5))
    for _ in range(12):
        x, y = rng.integers(0, max(1, w - 150), 2)
        d.ellipse([x, y, x + 130, y + 130], fill=tuple(rng.integers(0, 255, 3).tolist()))
    arr = np.asarray(img).astype(np.int16)
    arr += rng.integers(-28, 28, size=arr.shape, dtype=np.int16)
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class FakeCandidate:
    def __init__(self, data: bytes, url: str, slot: str):
        self.data, self.url, self.slot = data, url, slot


def make_validator() -> V.ReferenceValidator:
    """Builds a validator without running __init__ (which needs a real model)."""
    return V.ReferenceValidator.__new__(V.ReferenceValidator)


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #

def test_blueprint_survives_ortho_but_dies_under_hero() -> None:
    """The core regression: photographic gates delete line drawings."""
    val = make_validator()
    data = blueprint()

    assert val.prefilter(data, ORTHO) is not None, \
        "blueprint should pass the ORTHO profile"
    assert val.prefilter(data, HERO) is None, \
        "blueprint is expected to fail the photographic HERO profile"


def test_white_gate_disabled_for_ortho() -> None:
    assert profile_for(ORTHO)["white_gate"] is None
    assert profile_for(HERO)["white_gate"] is not None


def test_ortho_negatives_exclude_white_background() -> None:
    joined = " ".join(profile_for(ORTHO)["negatives"]).lower()
    assert "white background" not in joined, \
        "ORTHO must not penalise white backgrounds"
    assert "white background" in " ".join(profile_for(HERO)["negatives"]).lower()


def test_laplacian_is_resolution_normalised() -> None:
    """The same picture at two sizes must land on the same side of the gate."""
    val = make_validator()
    small = Image.open(io.BytesIO(photo(700, 700, seed=1)))
    big = small.resize((2800, 2800), Image.Resampling.LANCZOS)

    b1, b2 = io.BytesIO(), io.BytesIO()
    small.save(b1, format="PNG")
    big.save(b2, format="PNG")

    assert (val.prefilter(b1.getvalue(), HERO) is not None) == \
           (val.prefilter(b2.getvalue(), HERO) is not None)


def test_dhash_separates_reuploads_from_distinct_images() -> None:
    """A recompressed re-upload must hash close; a different image must not."""
    a = Image.open(io.BytesIO(photo(seed=11)))
    buf = io.BytesIO()
    a.save(buf, format="JPEG", quality=70)
    a_jpeg = Image.open(io.BytesIO(buf.getvalue()))
    b = Image.open(io.BytesIO(photo(seed=12)))

    same = int(np.count_nonzero(V._dhash(a) != V._dhash(a_jpeg)))
    diff = int(np.count_nonzero(V._dhash(a) != V._dhash(b)))

    assert same <= V.DHASH_DUPE_DISTANCE, f"re-upload hashed {same} bits apart"
    assert diff > V.DHASH_DUPE_DISTANCE * 2, f"distinct images only {diff} bits apart"


def test_dhash_removes_reuploads() -> None:
    """Same image at two URLs should survive once."""
    val = make_validator()
    data = photo(seed=7)
    recompressed = io.BytesIO()
    Image.open(io.BytesIO(data)).save(recompressed, format="JPEG", quality=70)

    cands = [
        FakeCandidate(data, "http://a/1.png", HERO),
        FakeCandidate(recompressed.getvalue(), "http://b/1.jpg", HERO),
        FakeCandidate(photo(seed=8), "http://c/2.png", HERO),
    ]
    survivors = val.prefilter_all(cands)
    assert len(survivors) == 2, f"expected 2 survivors, got {len(survivors)}"


def _scored(slot: str, n: int, base: float) -> List[V.ScoredImage]:
    img = Image.new("RGB", (10, 10))
    return [
        V.ScoredImage(image=img, url=f"http://x/{slot}{i}", slot=slot,
                      score=base - i * 0.001)
        for i in range(n)
    ]


def test_quota_guarantees_every_slot() -> None:
    """
    The headline bug. Hero images outscore everything; a global top-12 returns
    only hero shots. The quota must still seat the other three slots.
    """
    scored = (
        _scored(HERO, 20, 0.90)      # dominates on raw score
        + _scored(ORTHO, 5, 0.10)
        + _scored(DETAIL, 5, 0.12)
        + _scored(MATERIAL, 5, 0.11)
    )

    naive = sorted(scored, key=lambda s: s.score, reverse=True)[:12]
    assert {s.slot for s in naive} == {HERO}, "sanity: naive top-N is all hero"

    board = select_board_ref(scored, 12)
    counts = {s: sum(1 for b in board if b.slot == s) for s in (HERO, ORTHO, DETAIL, MATERIAL)}
    assert len(board) == 12
    assert all(c == 3 for c in counts.values()), counts


def test_quota_backfills_when_a_slot_is_empty() -> None:
    """No blueprints available should still return a full board."""
    scored = _scored(HERO, 20, 0.9) + _scored(DETAIL, 20, 0.5)
    board = select_board_ref(scored, 12)
    assert len(board) == 12
    assert {s.slot for s in board} == {HERO, DETAIL}


def test_quota_backfills_when_a_slot_is_short() -> None:
    scored = _scored(HERO, 20, 0.9) + _scored(ORTHO, 1, 0.2) + _scored(DETAIL, 20, 0.5)
    board = select_board_ref(scored, 12)
    assert len(board) == 12
    assert sum(1 for b in board if b.slot == ORTHO) == 1, \
        "the one available blueprint must be kept"


def test_output_is_grouped_by_slot() -> None:
    scored = _scored(HERO, 5, 0.9) + _scored(ORTHO, 5, 0.2) + _scored(MATERIAL, 5, 0.3)
    board = select_board_ref(scored, 12)
    slots = [b.slot for b in board]
    # each slot's entries must be contiguous
    for slot in set(slots):
        idx = [i for i, s in enumerate(slots) if s == slot]
        assert idx == list(range(idx[0], idx[-1] + 1)), f"{slot} is not contiguous: {slots}"


def test_short_board_returns_what_exists() -> None:
    board = select_board_ref(_scored(HERO, 2, 0.9), 12)
    assert len(board) == 2


select_board_ref = V.select_board


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
