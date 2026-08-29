import os
import sys
import tempfile
import subprocess
from typing import Any
from PIL import Image

# Ensure 'backend' directory is in Python path for purformat internal imports
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    import pureref_gen
except ImportError:
    from backend import pureref_gen

def build_and_launch_pur(images: list[Image.Image], output_path: str) -> str:
    """Saves verified images to a temporary buffer, packs them with pureref_gen, and launches PureRef."""
    # 1. Create a temporary folder to stage the clean PNGs
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, img in enumerate(images):
            # Scale down massive images for canvas balance
            target_w, target_h = img.size
            if target_w > 1200 or target_h > 1200:
                scale = 1200 / max(target_w, target_h)
                img = img.resize((int(target_w * scale), int(target_h * scale)), Image.Resampling.LANCZOS)

            temp_img_path = os.path.join(temp_dir, f"ref_{i:02d}.png")
            img.save(temp_img_path, format="PNG", optimize=True)

        # 2. Let pureref_gen build the complete PureRef canvas
        gen_fn: Any = getattr(pureref_gen, "pureref_gen", None) or getattr(pureref_gen, "generate", None)
        if callable(gen_fn):
            gen_fn(temp_dir, output_path)
        else:
            raise RuntimeError("Could not find generator function in pureref_gen module.")

    print(f"[ARCA Packer] Successfully compiled PureRef canvas at: {output_path}")

    # 3. Auto-open on macOS if PureRef is installed
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", output_path])
        except Exception as e:
            print(f"[Warning] Auto-launch skipped: {e}")

    return output_path