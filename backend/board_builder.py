"""
Builds a .pur canvas directly from in-memory images.

The row-packing layout below is adapted from the `pureref_gen.py` example in the
purformat project (see NOTICE). It is reimplemented here rather than called
directly for two reasons:

  * the original reads from a folder, which meant every image was written to
    disk as a PNG and then immediately re-opened and re-encoded as a PNG;
  * it sets each item's name and source to the file path it read from, which for
    us is a temporary directory that will not exist by the time the user opens
    the board. Source URLs are far more useful, and we have them.
"""

import os
import subprocess
import sys
from io import BytesIO
from typing import List, Sequence, Tuple

from PIL import Image

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import purformat.items as items
from purformat import purformat

# Images larger than this on their long edge are downscaled before packing, to
# keep the .pur file to a sane size.
MAX_EDGE = 1400

ROW_HEIGHT = 1000
ROW_TARGET_WIDTH = 2000


def _to_pur_image(img: Image.Image, name: str, source: str):
    """Converts one PIL image into a PurImage with a single transform."""
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                         Image.Resampling.LANCZOS)

    rgb = img.convert("RGB")
    pur_image = items.PurImage()
    with BytesIO() as buf:
        rgb.save(buf, format="PNG", compress_level=7)
        pur_image.pngBinary = buf.getvalue()

    transform = items.PurGraphicsImageItem()
    transform.reset_crop(rgb.width, rgb.height)
    transform.name = name
    transform.source = source
    pur_image.transforms = [transform]
    return pur_image


def _pack_rows(transforms: List) -> None:
    """Splits transforms into rows and assigns canvas coordinates in place."""
    for t in transforms:
        t.scale_to_height(ROW_HEIGHT)

    total_width = sum(t.width for t in transforms)

    rows = [transforms]
    while len(rows) * float(ROW_TARGET_WIDTH) < total_width:
        total_width /= 2.0
        new_rows = []
        for row in rows:
            remaining = total_width
            middle_index = 0
            while remaining > 0 and middle_index < len(row):
                remaining -= row[middle_index].width
                middle_index += 1
            new_rows.append(row[:middle_index])
            new_rows.append(row[middle_index:])
        rows = new_rows

    placement_y = 0.0
    for row in [r for r in rows if r]:
        row_width = sum(t.width for t in row)
        if row_width <= 0:
            continue
        scale_factor = 1000 / row_width

        placement_x = 0.0
        for t in row:
            t.scale(scale_factor)
            t.x = placement_x + t.width / 2
            placement_x += t.width
            t.y = placement_y + t.height / 2

        placement_y += ROW_HEIGHT * scale_factor


def build_board(
    entries: Sequence[Tuple[Image.Image, str, str]], output_path: str
) -> str:
    """
    Writes a .pur file from (image, slot_label, source_url) entries.

    Entries are packed in the order given, so the caller controls grouping.
    """
    if not entries:
        raise ValueError("No images to write.")

    pur_file = purformat.PurFile()
    pur_file.images = [
        _to_pur_image(img, name=f"{label} - {url}"[:200], source=url)
        for img, label, url in entries
    ]

    _pack_rows([t for image in pur_file.images for t in image.transforms])

    pur_file.write(output_path)
    print(f"[Board] Wrote {len(entries)} images to {output_path}")
    return output_path


def launch_board(path: str) -> None:
    """Opens the board in the system's default .pur handler, if there is one."""
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif sys.platform == "win32":
            # Previously missing: Windows users got no auto-launch at all.
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception as exc:
        print(f"[Board] Auto-launch skipped: {exc}")
