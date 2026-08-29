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