import io
import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from slots import SLOT_ORDER, profile_for

# Bump when filtering behaviour changes. Printed at startup and served from
# /health, so "is the new code actually running" is answerable in one look.
BUILD = "2026.08.29-subject-gate"

# Laplacian variance scales with image resolution, so the same threshold means
# different things for a 600px thumbnail and a 4000px scan. Every image is
# downscaled to this working size before the sharpness check so the number is
# comparable across candidates.
LAPLACIAN_WORK_DIM = 800

# Two images whose normalised CLIP embeddings exceed this cosine similarity are
# treated as the same picture. Catches re-uploads, crops and recompressions that
# a URL-level set() cannot.
EMBED_DUPE_THRESHOLD = 0.94

# Hamming distance between 64-bit difference hashes below which two images are
# treated as identical. Cheap pre-pass so obvious repeats never reach CLIP.
DHASH_DUPE_DISTANCE = 5

CLIP_BATCH_SIZE = 16

# How far below its slot's best subject match an image may fall before it is
# dropped. This is deliberately RELATIVE rather than an absolute floor: line
# drawings score systematically lower against a subject anchor than photographs
# do, so a single global threshold would delete the orthographic slot all over
# again. Comparing within a slot adapts to that automatically.
SUBJECT_MARGIN = float(os.environ.get("ASSEMBLAGE_SUBJECT_MARGIN", "0.045"))

# Optional hard floor on subject similarity, applied on top of the relative
# margin. Off by default because the right value depends on the CLIP score
# distribution for your prompts, which has to be measured rather than guessed:
# run tools/inspect.py to see real numbers, then set this.
SUBJECT_FLOOR = float(os.environ.get("ASSEMBLAGE_SUBJECT_FLOOR", "0.0"))

# Set ASSEMBLAGE_VERBOSE=1 to print a per-image table of scores and drop reasons.
VERBOSE = os.environ.get("ASSEMBLAGE_VERBOSE", "").strip().lower() in ("1","true","yes")


class ScoredImage(NamedTuple):
    image: Image.Image
    url: str
    slot: str
    score: float
    subject: float = 0.0


def _dhash(img: Image.Image, size: int = 8) -> np.ndarray:
    """64-bit difference hash as a bool array. No extra dependency needed."""
    small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten()


class ReferenceValidator:
    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device: str = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[Validator] build {BUILD}")
        print(f"[Validator] subject margin {SUBJECT_MARGIN} floor {SUBJECT_FLOOR}")
        print(f"[Validator] Initialising OpenCLIP ViT-B-32 on device: {self.device}")

        model_obj, _, preprocess_fn = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.model: Any = model_obj.to(self.device).eval()
        self.preprocess: Any = preprocess_fn
        self.tokenizer: Any = open_clip.get_tokenizer("ViT-B-32")

        # Negative anchors are prompt-independent, so they are encoded once at
        # startup rather than once per image. Each slot gets its own set because
        # the white-background negative is correct for photographs and wrong for
        # line drawings.
        self.neg_feats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for slot in SLOT_ORDER:
                prompts = profile_for(slot)["negatives"]
                tokens = self.tokenizer(prompts).to(self.device)
                feats: Any = self.model.encode_text(tokens)
                self.neg_feats[slot] = feats / feats.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ #
    # Stage 1: cheap per-image gates
    # ------------------------------------------------------------------ #

    def prefilter(self, image_bytes: bytes, slot: str) -> Optional[Image.Image]:
        """
        Applies the geometric and classical-CV gates for this image's slot.

        Returns the decoded image if it survives, else None. Every threshold
        here comes from the slot profile, because the photographic defaults
        delete orthographic plates: they are mostly white paper and have low
        global edge variance.
        """
        profile = profile_for(slot)

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None

        w, h = img.size
        if w < profile["min_dim"] or h < profile["min_dim"]:
            return None
        if max(w / h, h / w) > profile["max_aspect"]:
            return None

        work = img
        if max(w, h) > LAPLACIAN_WORK_DIM:
            scale = LAPLACIAN_WORK_DIM / max(w, h)
            work = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                              Image.Resampling.LANCZOS)

        gray = np.array(work.convert("L"))

        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < profile["min_laplacian"]:
            return None

        white_gate = profile["white_gate"]
        if white_gate is not None:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            white_ratio = float(hist[250:].sum() / gray.size)
            if white_ratio > white_gate:
                return None

        return img

    def prefilter_all(
        self, candidates: Sequence
    ) -> List[Tuple[Image.Image, str, str]]:
        """
        Runs the cheap gates over every candidate and removes exact repeats.

        Candidates are expected to expose .data, .url and .slot. The difference
        hash pass here catches the same picture re-hosted at a second URL, which
        the URL-level dedup in the scraper cannot see. Doing it before CLIP
        means duplicates cost nothing to reject.
        """
        survivors: List[Tuple[Image.Image, str, str]] = []
        hashes: List[np.ndarray] = []
        dupes = 0

        for cand in candidates:
            img = self.prefilter(cand.data, cand.slot)
            if img is None:
                continue

            h = _dhash(img)
            if any(int(np.count_nonzero(h != prev)) <= DHASH_DUPE_DISTANCE for prev in hashes):
                dupes += 1
                continue

            hashes.append(h)
            survivors.append((img, cand.url, cand.slot))

        print(
            f"[Validator] {len(survivors)} of {len(candidates)} passed the gates "
            f"({dupes} exact duplicates removed)."
        )
        return survivors

    # ------------------------------------------------------------------ #
    # Stage 2: batched semantic scoring
    # ------------------------------------------------------------------ #

    def _encode_positives(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Encodes each slot's positive anchors once per generation.

        The old code re-encoded three text prompts inside the per-image loop,
        costing 3 x N text forward passes per board for no benefit.
        """
        feats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for slot in SLOT_ORDER:
                prompts = [t.format(prompt=prompt) for t in profile_for(slot)["positives"]]
                tokens = self.tokenizer(prompts).to(self.device)
                f: Any = self.model.encode_text(tokens)
                feats[slot] = f / f.norm(dim=-1, keepdim=True)
        return feats

    def _encode_subject(self, prompt: str) -> torch.Tensor:
        """
        Encodes the bare subject, with no slot styling attached.

        The slot positives describe both a subject and a *kind* of image
        ("orthographic technical drawing of X"). An image can match the kind
        strongly while having nothing to do with the subject, which is how a
        photograph of a castle spire ends up on a board about ceiling tiles.
        This anchor measures subject match alone.
        """
        with torch.no_grad():
            tokens = self.tokenizer([prompt, f"a photograph of {prompt}"]).to(self.device)
            f: Any = self.model.encode_text(tokens)
            return f / f.norm(dim=-1, keepdim=True)

    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Encodes images in batches rather than one forward pass each."""
        chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(images), CLIP_BATCH_SIZE):
                batch = images[start:start + CLIP_BATCH_SIZE]
                tensor = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
                feats: Any = self.model.encode_image(tensor)
                chunks.append(feats / feats.norm(dim=-1, keepdim=True))
        return torch.cat(chunks, dim=0)

    def score_candidates(
        self, survivors: Sequence[Tuple[Image.Image, str, str]], prompt: str
    ) -> List[ScoredImage]:
        """
        Scores prefiltered images against their own slot's anchors.

        survivors is a sequence of (image, url, slot).
        """
        if not survivors:
            return []

        images = [s[0] for s in survivors]
        img_feats = self._encode_images(images)
        pos_feats = self._encode_positives(prompt)
        subj_feats = self._encode_subject(prompt)

        # Embedding-level dedup. Greedy: walk in order and drop anything too
        # close to something already kept.
        keep_indices: List[int] = []
        for i in range(img_feats.shape[0]):
            if not keep_indices:
                keep_indices.append(i)
                continue
            kept = img_feats[keep_indices]
            if float((img_feats[i] @ kept.T).max().item()) < EMBED_DUPE_THRESHOLD:
                keep_indices.append(i)

        dropped = img_feats.shape[0] - len(keep_indices)
        if dropped:
            print(f"[Validator] Dropped {dropped} near-duplicate images.")

        # Subject relevance, measured per image and compared within its slot.
        subject_sim: Dict[int, float] = {
            i: float((img_feats[i] @ subj_feats.T).mean().item()) for i in keep_indices
        }
        best_in_slot: Dict[str, float] = {}
        for i in keep_indices:
            slot = survivors[i][2]
            best_in_slot[slot] = max(best_in_slot.get(slot, -1.0), subject_sim[i])

        scored: List[ScoredImage] = []
        report: List[tuple] = []

        for i in keep_indices:
            img, url, slot = survivors[i]
            profile = profile_for(slot)
            feat = img_feats[i]
            subj = subject_sim[i]
            pos = float((feat @ pos_feats[slot].T).mean().item())
            neg = float((feat @ self.neg_feats[slot].T).mean().item())
            score = pos - 0.75 * neg

            reason = ""
            # Relative gate: far off-subject compared with the best this slot
            # found. Relative rather than absolute because line drawings sit
            # lower against a subject anchor than photographs do.
            if subj < best_in_slot[slot] - SUBJECT_MARGIN:
                reason = f"off-subject (best {best_in_slot[slot]:.3f})"
            elif SUBJECT_FLOOR and subj < SUBJECT_FLOOR:
                reason = "below subject floor"
            elif score < profile["min_score"]:
                reason = f"style score < {profile['min_score']}"

            report.append((slot, subj, score, reason, url))
            if not reason:
                scored.append(ScoredImage(image=img, url=url, slot=slot,
                                          score=score, subject=subj))

        dropped = sum(1 for r in report if r[3])
        print(f"[Validator] Kept {len(scored)}, dropped {dropped} on subject/style.")

        if VERBOSE:
            print(f"{'slot':<9}{'subject':>8}{'style':>8}  {'verdict':<34}url")
            for slot, subj, score, reason, url in sorted(report, key=lambda r: (r[0], -r[1])):
                print(f"{slot:<9}{subj:>8.3f}{score:>8.3f}  "
                      f"{(reason or 'KEPT'):<34}{url[:70]}")

        return scored


# ---------------------------------------------------------------------- #
# Stage 3: quota selection
# ---------------------------------------------------------------------- #

def select_board(
    scored: Sequence[ScoredImage], target_count: int
) -> List[ScoredImage]:
    """
    Picks the final board with a per-slot quota instead of a global top-N.

    This is the fix for the central bug. Sorting every candidate by score and
    slicing the top twelve returns twelve hero shots, because a photograph of
    the subject scores higher against a photographic anchor than a blueprint
    does against anything. Guaranteeing each slot its share is the only way the
    board actually contains the cross-section the tool promises.

    Slots that cannot fill their quota release their unused places back to a
    shared pool, so a subject with no available blueprints still returns a full
    board rather than a short one.
    """
    if not scored:
        return []

    by_slot: Dict[str, List[ScoredImage]] = {slot: [] for slot in SLOT_ORDER}
    for item in scored:
        by_slot.setdefault(item.slot, []).append(item)
    for slot in by_slot:
        by_slot[slot].sort(key=lambda s: s.score, reverse=True)

    active_slots = [s for s in SLOT_ORDER if by_slot.get(s)]
    if not active_slots:
        return sorted(scored, key=lambda s: s.score, reverse=True)[:target_count]

    base = target_count // len(active_slots)
    remainder = target_count % len(active_slots)

    selected: List[ScoredImage] = []
    leftovers: List[ScoredImage] = []

    for idx, slot in enumerate(active_slots):
        quota = base + (1 if idx < remainder else 0)
        pool = by_slot[slot]
        selected.extend(pool[:quota])
        leftovers.extend(pool[quota:])

    # Backfill any places the quota could not fill, best score first.
    if len(selected) < target_count:
        leftovers.sort(key=lambda s: s.score, reverse=True)
        selected.extend(leftovers[:target_count - len(selected)])

    # Group by slot in the output order so the board reads as four categories
    # rather than a shuffle. The layout engine places images sequentially, so
    # ordering here is what produces visual grouping on the canvas.
    slot_rank = {slot: i for i, slot in enumerate(SLOT_ORDER)}
    selected.sort(key=lambda s: (slot_rank.get(s.slot, 99), -s.score))
    return selected