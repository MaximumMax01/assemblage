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





# --------------------------------------------------------------------- #
# Regressions from the "office ceiling tiles" board
# --------------------------------------------------------------------- #

def test_stock_hosts_blocked_by_hostname_not_substring() -> None:
    import scraper
    assert scraper.is_stock_host("https://thumbs.dreamstime.com/b/x.jpg")
    assert scraper.is_stock_host("https://as1.ftcdn.net/v2/jpg/1000_F.jpg")
    assert scraper.is_stock_host("https://www.shutterstock.com/image-photo/x.jpg")
    assert not scraper.is_stock_host("https://upload.wikimedia.org/a/Ceiling.jpg")
    # a path containing a blocked name must not trip the filter
    assert not scraper.is_stock_host("https://example.com/notshutterstock.com.jpg")


def test_subject_margin_is_relative_not_absolute() -> None:
    """
    An absolute subject floor would delete the ortho slot again, because line
    drawings sit lower against a subject anchor than photographs do.
    """
    assert isinstance(V.SUBJECT_MARGIN, float)
    assert 0 < V.SUBJECT_MARGIN < 0.2


def test_query_templates_keep_subject_dominant() -> None:
    """Nine-word queries let the modifiers outrank the subject in search."""
    import taxonomy
    for name, data in taxonomy.ARCHETYPE_DEFINITIONS.items():
        for slot, tpl in data["templates"].items():
            tail = tpl.replace("{prompt}", "").split()
            assert len(tail) <= 4, f"{name}/{slot} has {len(tail)} modifier words: {tpl}"



def _si(slot, subj, style, gated=False):
    return V.ScoredImage(image=Image.new("RGB", (8, 8)), url=f"{slot}:{subj}",
                         slot=slot, score=style, subject=subj, gated=gated)


def test_gated_images_fill_a_short_slot_rather_than_shrinking_the_board() -> None:
    """
    A slot with only two clean candidates must still reach its quota by pulling
    a gated one back, instead of handing its place to another slot.
    """
    scored = (
        [_si(HERO, 0.34, 0.25) for _ in range(6)]
        + [_si(DETAIL, 0.30, 0.22), _si(DETAIL, 0.29, 0.19),
           _si(DETAIL, 0.25, 0.14, gated=True)]
        + [_si(ORTHO, 0.25, 0.24) for _ in range(4)]
        + [_si(MATERIAL, 0.35, 0.23) for _ in range(4)]
    )
    board = V.select_board(scored, 12)
    counts = {s: sum(1 for b in board if b.slot == s) for s in (HERO, ORTHO, DETAIL, MATERIAL)}
    assert counts == {HERO: 3, ORTHO: 3, DETAIL: 3, MATERIAL: 3}, counts
    assert sum(1 for b in board if b.gated) == 1


def test_gated_images_never_displace_clean_ones() -> None:
    """A gated image must lose to any ungated one in the same slot."""
    scored = [
        _si(HERO, 0.40, 0.40, gated=True),   # highest combined, but gated
        _si(HERO, 0.20, 0.10),
        _si(HERO, 0.19, 0.10),
        _si(HERO, 0.18, 0.10),
    ]
    board = V.select_board(scored, 3)
    assert not any(b.gated for b in board), "gated image displaced a clean one"


def test_combined_score_weights_subject_over_style() -> None:
    """
    Wrong subject, great style must lose to right subject, mediocre style --
    this is the mitre-joint failure mode.
    """
    right_subject = _si(DETAIL, 0.30, 0.10)
    wrong_subject = _si(DETAIL, 0.19, 0.22)
    assert right_subject.combined > wrong_subject.combined


def test_ai_generated_hosts_blocked() -> None:
    """
    AI renders are structurally wrong reference -- plausible-looking geometry
    that does not actually assemble. Both of these reached real boards.
    """
    import scraper
    assert scraper.is_ai_generated_host("https://imgcdn.stablediffusionweb.com/x.png")
    assert scraper.is_ai_generated_host("https://easy-peasy.ai/cdn-cgi/image/x.png")
    assert not scraper.is_ai_generated_host("https://upload.wikimedia.org/a/Porch.jpg")
    # must not be confused with the stock-photo list
    assert not scraper.is_ai_generated_host("https://thumbs.dreamstime.com/x.jpg")


# --------------------------------------------------------------------- #
# Floor-plan mode, hints, and edited searches
# --------------------------------------------------------------------- #

import taxonomy as T  # noqa: E402
import slots as S     # noqa: E402


def test_floor_plan_prompts_switch_mode_and_strip_the_view_type() -> None:
    assert T.detect_mode("Floor plan hotel") == (S.PLAN_MODE, "hotel")
    assert T.detect_mode("floor plan of a small office") == (S.PLAN_MODE, "small office")
    assert T.detect_mode("backrooms office floorplans") == (S.PLAN_MODE, "backrooms office")
    assert T.detect_mode("office ceiling tiles")[0] == S.OBJECT_MODE


def test_every_mode_slot_is_fully_defined() -> None:
    for name, mode in S.MODES.items():
        assert len(mode["slots"]) == 4, name
        for tpl in mode["subject_templates"]:
            assert "{prompt}" in tpl
        for slot in mode["slots"]:
            p = S.profile_for(slot)
            assert p["label"] and p["description"], slot
            assert all("{prompt}" in t for t in p["positives"]), slot
            assert S.mode_of(slot) == name


def test_plan_slots_never_white_gate() -> None:
    """Plans are drawings; drawings are mostly white paper. Same lesson as ORTHO."""
    for slot in S.slots_for(S.PLAN_MODE):
        assert S.profile_for(slot)["white_gate"] is None, slot


def test_plan_templates_keep_subject_dominant() -> None:
    for slot, tpl in T.PLAN_TEMPLATES.items():
        assert len(tpl.replace("{prompt}", "").split()) <= 4, tpl


def test_profile_for_rejects_unknown_slots() -> None:
    try:
        S.profile_for("hero_typo")
    except ValueError:
        return
    raise AssertionError("unknown slot silently accepted")


def test_select_board_applies_quota_to_plan_slots() -> None:
    """
    Regression: plan slots were missing from SLOT_ORDER, so select_board skipped
    them all and fell back to a global top-N with no quota.
    """
    order = S.slots_for(S.PLAN_MODE)
    scored = (
        [_si(order[0], 0.35, 0.30) for _ in range(10)]    # dominates raw score
        + [_si(order[1], 0.20, 0.10) for _ in range(4)]
        + [_si(order[2], 0.21, 0.11) for _ in range(4)]
        + [_si(order[3], 0.22, 0.12) for _ in range(4)]
    )
    board = V.select_board(scored, 12, order)
    counts = [sum(1 for b in board if b.slot == sl) for sl in order]
    assert counts == [3, 3, 3, 3], counts
    # and without an explicit order it must still not drop them
    board2 = V.select_board(scored, 12)
    assert [sum(1 for b in board2 if b.slot == sl) for sl in order] == [3, 3, 3, 3]


def test_mood_words_get_concrete_suggestions() -> None:
    mode, subj = T.detect_mode("backrooms floor plan")
    hints = T.prompt_hints("backrooms floor plan", mode, subj)
    assert hints and hints[0]["kind"] == "mood"
    assert "tenant improvement floor plan" in hints[0]["suggestions"]


def test_scene_hint_uses_head_noun_only() -> None:
    """'office ceiling tiles' is about tiles; 'modern office' is about a room."""
    def kinds(p):
        m, s = T.detect_mode(p)
        return [h["kind"] for h in T.prompt_hints(p, m, s)]
    assert "scene" not in kinds("office ceiling tiles")
    assert "scene" in kinds("modern office")


def test_exterior_scenes_do_not_suggest_floor_plans() -> None:
    m, subj = T.detect_mode("house front")
    hint = T.prompt_hints("house front", m, subj)[0]
    assert hint["kind"] == "scene" and hint["suggestions"] == []


def test_overrides_validate_and_blank_switches_a_slot_off() -> None:
    mode, qs = T.resolve_overrides([("hero", "oak chair"), ("ortho", "   "),
                                    ("detail", "oak chair joint"), ("material", "oak grain")])
    assert mode == S.OBJECT_MODE
    assert [q.slot for q in qs] == ["hero", "detail", "material"]

    bad = [
        [("hero", "x"), ("plan_simple", "y")],     # mixed modes
        [("hero", "x"), ("hero", "y")],            # duplicate slot
        [("nonsense", "x")],                       # unknown slot
        [("hero", ""), ("ortho", " ")],            # everything blank
        [("hero", "x" * 201)],                     # too long
    ]
    for case in bad:
        try:
            T.resolve_overrides(case)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid overrides: {case}")


# --------------------------------------------------------------------- #
# Light fixtures, and the board layout
# --------------------------------------------------------------------- #

def test_light_nouns_route_to_fixture_archetype_by_head_noun() -> None:
    assert T.route_archetype("stage light") == "lighting_fixture"
    assert T.route_archetype("ceiling stage light") == "lighting_fixture"
    assert T.route_archetype("fluorescent troffer light") == "lighting_fixture"
    assert T.route_archetype("brass chandelier") == "lighting_fixture"
    # "light" as an adjective or a modifier is not a light
    assert T.route_archetype("light oak table") is None
    assert T.route_archetype("light switch") is None


def test_stage_light_plan_asks_for_the_fixture_not_the_glow() -> None:
    engine = T.TaxonomyEngine.__new__(T.TaxonomyEngine)   # routing needs no model
    plan = engine.plan("stage light")
    assert plan.archetype == "lighting_fixture"
    assert plan.kind_label == "Light fixture"
    assert all("stage light" in q.query for q in plan.queries)
    assert any("fixture" in q.query for q in plan.queries)
    assert plan.extra_negatives, "fixture archetype must push against the lit-stage sense"
    assert all("{prompt}" in t for t in plan.subject_templates)


def test_every_archetype_has_a_label() -> None:
    for name, data in T.ARCHETYPE_DEFINITIONS.items():
        assert data.get("label"), name


def _layout(group_aspects):
    import board_builder as BB
    import purformat.items as items
    groups = []
    for aspects in group_aspects:
        g = []
        for a in aspects:
            t = items.PurGraphicsImageItem()
            t.reset_crop(int(1000 * a), 1000)
            g.append(t)
        groups.append(g)
    BB._pack_rows(groups)
    return BB, groups


def test_layout_never_enlarges_a_short_row() -> None:
    """
    Regression: the old layout stretched every row to full width, so a single
    leftover image came out about 3.5x the height of everything else.
    """
    BB, groups = _layout([[1.5, 1.0, 1.1, 1.0], [1.3, 1.3, 2.0], [1.0, 1.6, 0.9, 1.9], [1.45]])
    heights = [t.height for g in groups for t in g]
    assert max(heights) <= BB.ROW_HEIGHT + 1e-6, max(heights)
    lone = groups[-1][0].height
    typical = sorted(heights)[len(heights) // 2]
    assert lone <= typical * 1.5, f"lone image {lone:.0f} vs typical {typical:.0f}"


def test_layout_keeps_groups_on_separate_rows_and_inside_the_canvas() -> None:
    BB, groups = _layout([[1.0, 1.0, 1.0], [1.3], [1.3, 2.0, 1.0, 1.6, 0.9], [1.9, 1.45, 1.0]])
    def top(t): return t.y - t.height / 2
    def bottom(t): return t.y + t.height / 2
    for a, b in zip(groups, groups[1:]):
        assert max(bottom(t) for t in a) < min(top(t) for t in b), "groups overlap vertically"
    for g in groups:
        for t in g:
            assert t.x - t.width / 2 >= -1e-6
            assert t.x + t.width / 2 <= BB.CANVAS_WIDTH + 1e-6

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