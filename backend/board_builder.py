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


# Layout, in PureRef canvas units. Every image starts at ROW_HEIGHT tall.
CANVAS_WIDTH = 4000
GAP = 40            # between images, and between rows within a group
GROUP_GAP = 200     # extra space between groups, so they read as separate


def _pack_rows(groups: List[List]) -> None:
    """
    Lays out images as a justified grid, one group after another.

    A full row is scaled so it spans exactly CANVAS_WIDTH. A row that cannot be
    filled -- a group's last few images -- is never scaled UP; it keeps the
    normal row height and simply ends early.

    The previous layout scaled every row to full width, including a final row
    holding a single leftover image. That image was blown up to roughly three
    and a half times the height of everything else, and since boards are
    ordered with the weakest leftovers last, the image inflated was usually the
    worst one on the board.

    Each group starts on a fresh row, so hero shots, drawings, details and
    textures never share a row.
    """
    y = 0.0
    for g_index, group in enumerate(groups):
        for t in group:
            t.scale_to_height(ROW_HEIGHT)

        rows: List[List] = []
        current: List = []
        for t in group:
            current.append(t)
            span = sum(x.width for x in current) + GAP * (len(current) - 1)
            if span >= CANVAS_WIDTH:          # row is full: close it
                rows.append(current)
                current = []
        if current:
            rows.append(current)

        for row in rows:
            content = sum(t.width for t in row)
            available = CANVAS_WIDTH - GAP * (len(row) - 1)
            # Full rows shrink to fit exactly. Short rows are capped at 1.0,
            # so they are never enlarged past normal height.
            scale = min(1.0, available / content) if content > 0 else 1.0

            x = 0.0
            height = 0.0
            for t in row:
                t.scale(scale)
                # PureRef positions an item by its centre, not its corner.
                t.x = x + t.width / 2
                t.y = y + t.height / 2
                x += t.width + GAP
                height = max(height, t.height)
            y += height + GAP

        if g_index < len(groups) - 1:
            y += GROUP_GAP


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

    # Consecutive entries with the same label form a group. select_board
    # already orders the board group by group, so this recovers the groups.
    groups: List[List] = []
    last_label = None
    for (img, label, url), image in zip(entries, pur_file.images):
        if label != last_label:
            groups.append([])
            last_label = label
        groups[-1].extend(image.transforms)
    _pack_rows(groups)

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
