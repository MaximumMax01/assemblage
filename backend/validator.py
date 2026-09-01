<<<<<<< HEAD
import io
from typing import Any, Tuple, Optional
import cv2
import numpy as np
import torch
import open_clip
from PIL import Image

class ReferenceValidator:
    def __init__(self) -> None:
        # Determine best available hardware accelerator on macOS
        if torch.backends.mps.is_available():
            self.device: str = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"[ARCA Core] Initializing OpenCLIP ViT-B-32 on device: {self.device}")
        
        # We type-hint as Any so Pylance knows these have dynamic methods
        model_obj, _, preprocess_fn = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.model: Any = model_obj.to(self.device).eval()
        self.preprocess: Any = preprocess_fn
        self.tokenizer: Any = open_clip.get_tokenizer('ViT-B-32')

        # Negative semantic anchors to filter out junk
        self.neg_prompts = [
            "e-commerce product listing on pure white background",
            "blurry low resolution thumbnail with jpeg artifacts",
            "text watermark meme user interface graphic",
            "video game inventory skin card market price listing",
            "counter strike trading market hud banner graphic"
        ]
        with torch.no_grad():
            neg_tokens = self.tokenizer(self.neg_prompts).to(self.device)
            neg_feats: Any = self.model.encode_text(neg_tokens)
            self.neg_feats: torch.Tensor = neg_feats / neg_feats.norm(dim=-1, keepdim=True)

    def validate_and_score(self, image_bytes: bytes, prompt: str) -> Tuple[bool, Optional[Image.Image], float]:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return False, None, 0.0

        w, h = img.size

        # 1. Geometric & Aspect Ratio Filter
        if w < 600 or h < 600 or max(w / h, h / w) > 3.2:
            return False, None, 0.0

        # 2. OpenCV Blur & Dynamic Range Filters
        cv_img = np.array(img.convert("L"))
        
        # A. Laplacian sharpness check
        laplacian_var = float(cv2.Laplacian(cv_img, cv2.CV_64F).var())
        if laplacian_var < 90.0:
            return False, None, 0.0

        # B. Pure white e-commerce cutout check
        hist = cv2.calcHist([cv_img], [0], None, [256], [0, 256])
        white_pixel_ratio = float(hist[250:].sum() / (w * h))
        if white_pixel_ratio > 0.70:
            return False, None, 0.0

        # 3. OpenCLIP Zero-Shot Semantic Relevance
        pos_prompts = [
            f"sharp high quality photograph of {prompt}",
            f"detailed technical reference of {prompt}",
            f"macro surface texture of {prompt}"
        ]

        with torch.no_grad():
            pos_tokens = self.tokenizer(pos_prompts).to(self.device)
            pos_feats: Any = self.model.encode_text(pos_tokens)
            pos_feats = pos_feats / pos_feats.norm(dim=-1, keepdim=True)

            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            img_feat: Any = self.model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

            pos_score = float((img_feat @ pos_feats.T).mean().item())
            neg_score = float((img_feat @ self.neg_feats.T).mean().item())
            final_quality = pos_score - (neg_score * 0.75)

        if final_quality < 0.05:
            return False, None, 0.0

        return True, img, final_quality
=======
# Import the in-memory byte stream library (handles raw image file bytes in RAM like a file)
import io
# Import the operating system module (for reading environment variables and filesystem paths)
import os
# Import the system module (for modifying Python's import search paths and system settings)
import sys
# Import type hint helpers so we can document data types like C structs/arrays (List, Tuple, Dict, etc.)
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

# Import OpenCV for classical computer vision image operations (edge detection, histograms)
import cv2
# Import NumPy for fast, C-speed multidimensional array math operations
import numpy as np
# Import PyTorch for running neural network tensors on hardware accelerators (CPU/GPU)
import torch
# Import OpenCLIP to load AI models that convert images and text into mathematical vector embeddings
import open_clip
# Import Pillow (PIL) for basic image loading, resizing, and color conversion
from PIL import Image

# Get the absolute folder path of the directory containing this script file
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if this backend folder is currently missing from Python's module lookup path list
if BACKEND_DIR not in sys.path:
    # Prepend this folder to index 0 of the search path so local imports resolve first
    sys.path.insert(0, BACKEND_DIR)

# Import the slot order list and slot-specific configuration profiles from the local slots.py file
from slots import SLOT_ORDER, profile_for

# Version build tag used to verify in logs or health checks which code revision is running
BUILD = "2026.08.29-subject-gate"

# Standard dimension (in pixels) to downscale images to before testing for blur/sharpness
LAPLACIAN_WORK_DIM = 800

# Cosine similarity threshold (0.0 to 1.0); AI vector similarity >= 0.94 is considered a duplicate image
EMBED_DUPE_THRESHOLD = 0.94

# Hamming distance threshold; difference hashes differing by <= 5 bits are treated as exact duplicates
DHASH_DUPE_DISTANCE = 5

# Number of images sent to the neural network simultaneously in a single forward pass
CLIP_BATCH_SIZE = 16

# Maximum allowed drop in subject score relative to the best candidate in the same category/slot
SUBJECT_MARGIN = float(os.environ.get("ASSEMBLAGE_SUBJECT_MARGIN", "0.045"))

# Optional absolute floor for subject similarity (disabled at 0.0 by default)
SUBJECT_FLOOR = float(os.environ.get("ASSEMBLAGE_SUBJECT_FLOOR", "0.0"))

# Boolean flag to enable verbose per-image diagnostic printing via environment variable
VERBOSE = os.environ.get("ASSEMBLAGE_VERBOSE", "").strip().lower() in ("1","true","yes")


# Define a lightweight immutable data structure (similar to a C struct) for a scored candidate image
class ScoredImage(NamedTuple):
    # The loaded Pillow RGB image object
    image: Image.Image
    # The source web URL where the image was scraped from
    url: str
    # The assigned board category/slot name (e.g. "hero_photo", "orthographic")
    slot: str
    # The combined stylistic quality score
    score: float
    # The semantic subject relevance score (defaults to 0.0)
    subject: float = 0.0


# Compute a 64-bit difference perceptual hash array to detect visually identical images
def _dhash(img: Image.Image, size: int = 8) -> np.ndarray:
    """64-bit difference hash as a bool array. No extra dependency needed."""
    # Convert image to grayscale ("L") and shrink to a 9x8 grid using high-quality Lanczos resampling
    small = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    # Convert the Pillow image into an 8x9 NumPy array of 16-bit integers (pixel intensities)
    arr = np.asarray(small, dtype=np.int16)
    # Compare each pixel to its right neighbor (col 1..8 > col 0..7) and flatten into a 64-boolean array
    return (arr[:, 1:] > arr[:, :-1]).flatten()


# Main validator class containing the filtering and scoring pipeline
class ReferenceValidator:
    # Constructor method that runs when ReferenceValidator() is instantiated
    def __init__(self) -> None:
        # Check if an NVIDIA GPU with CUDA drivers is available
        if torch.cuda.is_available():
            # Select CUDA as the compute device
            self.device: str = "cuda"
        # Otherwise, check if Apple Silicon GPU (Metal Performance Shaders) is available
        elif torch.backends.mps.is_available():
            # Select MPS as the compute device
            self.device = "mps"
        # Otherwise, fall back to running on the standard CPU
        else:
            # Select CPU as the compute device
            self.device = "cpu"

        # Print the validator build revision to standard output
        print(f"[Validator] build {BUILD}")
        # Print the active subject margin and subject floor thresholds
        print(f"[Validator] subject margin {SUBJECT_MARGIN} floor {SUBJECT_FLOOR}")
        # Print the selected hardware accelerator device
        print(f"[Validator] Initialising OpenCLIP ViT-B-32 on device: {self.device}")

        # Load the OpenCLIP Vision Transformer model architecture, weights, and image preprocessing pipeline
        model_obj, _, preprocess_fn = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        # Move the neural network to the target device (GPU/CPU) and set it to evaluation/read-only mode
        self.model: Any = model_obj.to(self.device).eval()
        # Save the image preprocessing function to self for later image batch preparation
        self.preprocess: Any = preprocess_fn
        # Instantiate the text tokenizer that converts text strings into integer token IDs
        self.tokenizer: Any = open_clip.get_tokenizer("ViT-B-32")

        # Dictionary to store pre-computed negative style prompt embeddings for each slot
        self.neg_feats: Dict[str, torch.Tensor] = {}
        # Disable gradient calculations to save memory and compute during inference
        with torch.no_grad():
            # Iterate through each defined slot category
            for slot in SLOT_ORDER:
                # Retrieve the list of negative text prompts for this slot from its profile
                prompts = profile_for(slot)["negatives"]
                # Convert the text prompts into token ID tensors and move them to the device (GPU/CPU)
                tokens = self.tokenizer(prompts).to(self.device)
                # Pass the tokens through the text encoder to get raw vector representations
                feats: Any = self.model.encode_text(tokens)
                # Normalize the vectors to unit length (length 1.0) for fast cosine similarity dot products
                self.neg_feats[slot] = feats / feats.norm(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ #
    # Stage 1: cheap per-image gates
    # ------------------------------------------------------------------ #

    # Filter a single image using fast, classical checks (size, aspect ratio, blur, white space)
    def prefilter(self, image_bytes: bytes, slot: str) -> Optional[Image.Image]:
        """
        Applies the geometric and classical-CV gates for this image's slot.

        Returns the decoded image if it survives, else None. Every threshold
        here comes from the slot profile, because the photographic defaults
        delete orthographic plates: they are mostly white paper and have low
        global edge variance.
        """
        # Fetch the configuration settings dictionary specific to this category/slot
        profile = profile_for(slot)

        # Try to decode the raw bytes into a valid RGB image object
        try:
            # Wrap bytes in a memory buffer and open as a 3-channel RGB image
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # If the bytes are corrupted or not a valid image format, catch the error
        except Exception:
            # Return None to reject and drop this candidate image
            return None

        # Unpack the width (w) and height (h) dimensions of the image in pixels
        w, h = img.size
        # Check if either width or height is smaller than the slot's minimum allowed dimension
        if w < profile["min_dim"] or h < profile["min_dim"]:
            # Drop the image if it is too small
            return None
        # Check if the aspect ratio (width/height or height/width) exceeds the maximum allowed stretch
        if max(w / h, h / w) > profile["max_aspect"]:
            # Drop the image if it is overly wide, tall, or a panoramic strip
            return None

        # Create a working copy reference for blur analysis
        work = img
        # If the image's longest side exceeds the benchmark dimension (800px), downscale it
        if max(w, h) > LAPLACIAN_WORK_DIM:
            # Calculate the scaling factor needed to fit within the 800px benchmark
            scale = LAPLACIAN_WORK_DIM / max(w, h)
            # Resize the working image proportionally using high-quality resampling
            work = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                              Image.Resampling.LANCZOS)

        # Convert the resized working image to a grayscale NumPy 2D array
        gray = np.array(work.convert("L"))

        # Run a Laplacian edge-detection filter and calculate edge variance (measure of sharpness)
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < profile["min_laplacian"]:
            # Drop the image if edge variance is below the threshold (image is too blurry)
            return None

        # Read the maximum allowable white pixel percentage for this slot (if configured)
        white_gate = profile["white_gate"]
        # If this slot has a white background limit rule defined
        if white_gate is not None:
            # Compute a 256-bin histogram of grayscale pixel intensities (0 to 255)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            # Calculate the ratio of nearly pure-white pixels (bins 250 to 255) over total pixels
            white_ratio = float(hist[250:].sum() / gray.size)
            # If the proportion of white background exceeds the gate threshold
            if white_ratio > white_gate:
                # Drop the image (e.g. rejects studio white-background stock photos if not wanted)
                return None

        # Return the valid, surviving decoded image object
        return img

    # Run cheap Stage 1 prefilters and fast difference-hash deduplication across all candidates
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
        # Initialize an empty list to store candidate tuples that pass all Stage 1 filters
        survivors: List[Tuple[Image.Image, str, str]] = []
        # Initialize an empty list to store 64-bit dHash fingerprints of accepted images
        hashes: List[np.ndarray] = []
        # Initialize a counter for duplicates removed during this pass
        dupes = 0

        # Loop through each candidate object provided by the scraper/collector
        for cand in candidates:
            # Apply geometric, blur, and white-space gates to this candidate's raw data
            img = self.prefilter(cand.data, cand.slot)
            # If the image failed any gate and returned None
            if img is None:
                # Skip to the next candidate
                continue

            # Compute the 64-bit perceptual difference hash of this surviving image
            h = _dhash(img)
            # Check if this hash is within <= 5 bit flips of any previously accepted image hash
            if any(int(np.count_nonzero(h != prev)) <= DHASH_DUPE_DISTANCE for prev in hashes):
                # Increment the duplicate counter
                dupes += 1
                # Skip and discard this duplicate image
                continue

            # Store the unique hash fingerprint for future comparisons
            hashes.append(h)
            # Add the surviving (Image, URL, slot_name) tuple to the survivors list
            survivors.append((img, cand.url, cand.slot))

        # Print how many candidates passed Stage 1 and how many duplicates were removed
        print(
            f"[Validator] {len(survivors)} of {len(candidates)} passed the gates "
            f"({dupes} exact duplicates removed)."
        )
        # Return the list of surviving candidate image tuples
        return survivors

    # ------------------------------------------------------------------ #
    # Stage 2: batched semantic scoring
    # ------------------------------------------------------------------ #

    # Encode positive stylistic text descriptions for each slot using the user prompt
    def _encode_positives(self, prompt: str) -> Dict[str, torch.Tensor]:
        """
        Encodes each slot's positive anchors once per generation.

        The old code re-encoded three text prompts inside the per-image loop,
        costing 3 x N text forward passes per board for no benefit.
        """
        # Initialize an empty dictionary mapping slot names to encoded text feature tensors
        feats: Dict[str, torch.Tensor] = {}
        # Disable neural network gradient tracking during text encoding
        with torch.no_grad():
            # Loop through each slot category name
            for slot in SLOT_ORDER:
                # Format each prompt template string with the user prompt (e.g. "blueprint of {prompt}")
                prompts = [t.format(prompt=prompt) for t in profile_for(slot)["positives"]]
                # Convert the formatted text strings into token IDs and move them to the device (GPU/CPU)
                tokens = self.tokenizer(prompts).to(self.device)
                # Pass the tokens through the CLIP text encoder
                f: Any = self.model.encode_text(tokens)
                # Normalize feature vectors to unit length and store in dictionary by slot name
                feats[slot] = f / f.norm(dim=-1, keepdim=True)
        # Return the dictionary of normalized positive text embeddings
        return feats

    # Encode the bare subject prompt without any styling modifiers
    def _encode_subject(self, prompt: str) -> torch.Tensor:
        """
        Encodes the bare subject, with no slot styling attached.

        The slot positives describe both a subject and a *kind* of image
        ("orthographic technical drawing of X"). An image can match the kind
        strongly while having nothing to do with the subject, which is how a
        photograph of a castle spire ends up on a board about ceiling tiles.
        This anchor measures subject match alone.
        """
        # Disable neural network gradient tracking during text encoding
        with torch.no_grad():
            # Tokenize both the raw prompt and "a photograph of [prompt]"
            tokens = self.tokenizer([prompt, f"a photograph of {prompt}"]).to(self.device)
            # Pass the tokens through the CLIP text encoder
            f: Any = self.model.encode_text(tokens)
            # Normalize the feature vectors to unit length and return them
            return f / f.norm(dim=-1, keepdim=True)

    # Encode a sequence of images through the CLIP vision model in batches
    def _encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Encodes images in batches rather than one forward pass each."""
        # Initialize an empty list to collect batched feature tensors
        chunks: List[torch.Tensor] = []
        # Disable gradient calculations during image inference
        with torch.no_grad():
            # Step through the images list in chunks of CLIP_BATCH_SIZE (16)
            for start in range(0, len(images), CLIP_BATCH_SIZE):
                # Slice the current batch of 16 (or fewer) images
                batch = images[start:start + CLIP_BATCH_SIZE]
                # Preprocess each image into a normalized tensor, stack into a single batch, and move to GPU
                tensor = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
                # Run the batch of images through the CLIP vision encoder
                feats: Any = self.model.encode_image(tensor)
                # Normalize each image vector to unit length and append the batch to chunks
                chunks.append(feats / feats.norm(dim=-1, keepdim=True))
        # Concatenate all batch chunks along dimension 0 into one combined (N, 512) tensor
        return torch.cat(chunks, dim=0)

    # Score surviving images against subject relevance and slot style anchors
    def score_candidates(
        self, survivors: Sequence[Tuple[Image.Image, str, str]], prompt: str
    ) -> List[ScoredImage]:
        """
        Scores prefiltered images against their own slot's anchors.

        survivors is a sequence of (image, url, slot).
        """
        # If no images survived Stage 1, return an empty list immediately
        if not survivors:
            return []

        # Extract only the Pillow Image objects from the survivors tuples
        images = [s[0] for s in survivors]
        # Encode all images into AI feature vectors (N x 512 matrix)
        img_feats = self._encode_images(images)
        # Encode the positive style text prompts for all slots
        pos_feats = self._encode_positives(prompt)
        # Encode the generic bare subject text prompt
        subj_feats = self._encode_subject(prompt)

        # Initialize a list of integer indices for images to keep after embedding-level deduplication
        keep_indices: List[int] = []
        # Loop over every image index from 0 to N-1
        for i in range(img_feats.shape[0]):
            # If no images have been kept yet, always keep the first one
            if not keep_indices:
                keep_indices.append(i)
                continue
            # Slice the feature vectors of all images that have already been kept
            kept = img_feats[keep_indices]
            # Compute cosine similarity via matrix multiply (@); if max similarity < 0.94, it is unique
            if float((img_feats[i] @ kept.T).max().item()) < EMBED_DUPE_THRESHOLD:
                # Keep this image index as it is sufficiently visually distinct
                keep_indices.append(i)

        # Calculate the count of near-duplicate images dropped in this pass
        dropped = img_feats.shape[0] - len(keep_indices)
        # If any near-duplicates were detected, log the count
        if dropped:
            print(f"[Validator] Dropped {dropped} near-duplicate images.")

        # Compute the mean cosine similarity between each kept image vector and the subject text vectors
        subject_sim: Dict[int, float] = {
            i: float((img_feats[i] @ subj_feats.T).mean().item()) for i in keep_indices
        }
        # Dictionary to track the highest subject similarity score found inside each slot
        best_in_slot: Dict[str, float] = {}
        # Loop over each kept image index to discover the best score per slot
        for i in keep_indices:
            # Extract the slot name from the survivor tuple
            slot = survivors[i][2]
            # Update the max subject score seen so far for this slot
            best_in_slot[slot] = max(best_in_slot.get(slot, -1.0), subject_sim[i])

        # Initialize a list to hold the finalized ScoredImage namedtuples
        scored: List[ScoredImage] = []
        # Initialize a list to hold diagnostic report tuples for logging
        report: List[tuple] = []

        # Loop through each kept image index to compute final scores and gating decisions
        for i in keep_indices:
            # Unpack the image object, original URL, and assigned slot name
            img, url, slot = survivors[i]
            # Retrieve the configuration profile for this slot
            profile = profile_for(slot)
            # Get the image's 512-dimensional normalized feature vector
            feat = img_feats[i]
            # Retrieve the precomputed subject similarity score for this image
            subj = subject_sim[i]
            # Compute mean cosine similarity against the slot's positive style text anchors
            pos = float((feat @ pos_feats[slot].T).mean().item())
            # Compute mean cosine similarity against the slot's negative style text anchors
            neg = float((feat @ self.neg_feats[slot].T).mean().item())
            # Calculate composite style score: positive style minus 75% of negative style
            score = pos - 0.75 * neg

            # Initialize an empty rejection reason string
            reason = ""
            # Check if subject score is too far below the best subject score found in this slot
            if subj < best_in_slot[slot] - SUBJECT_MARGIN:
                # Mark as off-subject relative to the slot's best candidate
                reason = f"off-subject (best {best_in_slot[slot]:.3f})"
            # Check if subject score is below the absolute subject floor (if floor is configured)
            elif SUBJECT_FLOOR and subj < SUBJECT_FLOOR:
                # Mark as below the absolute subject floor
                reason = "below subject floor"
            # Check if the combined style score is below the slot's minimum threshold
            elif score < profile["min_score"]:
                # Mark as failing the style score requirement
                reason = f"style score < {profile['min_score']}"

            # Append this candidate's details and verdict to the diagnostic report list
            report.append((slot, subj, score, reason, url))
            # If no rejection reason was triggered (the image passed all gates)
            if not reason:
                # Instantiate a ScoredImage tuple and add it to the passing list
                scored.append(ScoredImage(image=img, url=url, slot=slot,
                                          score=score, subject=subj))

        # Count how many images were dropped during this semantic gate check
        dropped = sum(1 for r in report if r[3])
        # Print summary of kept vs dropped images
        print(f"[Validator] Kept {len(scored)}, dropped {dropped} on subject/style.")

        # If verbose mode is enabled, print a formatted table of all evaluated candidates
        if VERBOSE:
            # Print table column headers
            print(f"{'slot':<9}{'subject':>8}{'style':>8}  {'verdict':<34}url")
            # Sort the report rows by slot name, then by descending subject score, and print each row
            for slot, subj, score, reason, url in sorted(report, key=lambda r: (r[0], -r[1])):
                # Print formatted row containing slot, subject score, style score, verdict, and truncated URL
                print(f"{slot:<9}{subj:>8.3f}{score:>8.3f}  "
                      f"{(reason or 'KEPT'):<34}{url[:70]}")

        # Return the final list of validated and scored images
        return scored


# ---------------------------------------------------------------------- #
# Stage 3: quota selection
# ---------------------------------------------------------------------- #

# Select the final balanced set of images across all categories using a quota system
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
    # If the scored candidates list is empty, return an empty list immediately
    if not scored:
        return []

    # Initialize a dictionary grouping candidate images by their slot name
    by_slot: Dict[str, List[ScoredImage]] = {slot: [] for slot in SLOT_ORDER}
    # Loop over every scored candidate image
    for item in scored:
        # Append the candidate into its corresponding slot list
        by_slot.setdefault(item.slot, []).append(item)
    # Loop over each slot list in the dictionary
    for slot in by_slot:
        # Sort candidates within each slot by style score in descending order (highest score first)
        by_slot[slot].sort(key=lambda s: s.score, reverse=True)

    # Identify which slots have at least one valid candidate image available
    active_slots = [s for s in SLOT_ORDER if by_slot.get(s)]
    # If no slots have candidates, fall back to returning the global top N candidates
    if not active_slots:
        return sorted(scored, key=lambda s: s.score, reverse=True)[:target_count]

    # Calculate the base integer quota of images allocated to each active slot
    base = target_count // len(active_slots)
    # Calculate the remainder count to distribute among the first few slots
    remainder = target_count % len(active_slots)

    # Initialize a list to hold candidates selected via their slot quota
    selected: List[ScoredImage] = []
    # Initialize a list to collect leftover candidates not picked in the initial quota
    leftovers: List[ScoredImage] = []

    # Iterate through each active slot along with its index (0, 1, 2, ...)
    for idx, slot in enumerate(active_slots):
        # Calculate this slot's quota (adds +1 extra slot if idx < remainder)
        quota = base + (1 if idx < remainder else 0)
        # Reference the sorted candidates available in this slot
        pool = by_slot[slot]
        # Take up to `quota` images from this slot and add them to the selected list
        selected.extend(pool[:quota])
        # Add any remaining images beyond the quota to the leftovers pool
        leftovers.extend(pool[quota:])

    # If some slots were empty/short and we have fewer images than target_count
    if len(selected) < target_count:
        # Sort all leftover candidates globally by score in descending order
        leftovers.sort(key=lambda s: s.score, reverse=True)
        # Backfill the missing spots by taking the highest-scoring leftovers
        selected.extend(leftovers[:target_count - len(selected)])

    # Create a mapping of slot name -> rank index (0, 1, 2...) based on predefined SLOT_ORDER
    slot_rank = {slot: i for i, slot in enumerate(SLOT_ORDER)}
    # Sort the final selected images primarily by slot display order, and secondarily by score descending
    selected.sort(key=lambda s: (slot_rank.get(s.slot, 99), -s.score))
    # Return the finalized, balanced, and ordered list of images for the moodboard
    return selected
>>>>>>> b3ed0a9 (second commit, for school)
