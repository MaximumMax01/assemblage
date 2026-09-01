import asyncio
import json
import os
import re
import sys
import tempfile
import uuid

# Ensure backend directory is in Python search path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Install the system trust store for TLS-inspecting networks. This must not be a
# bare top-level import: truststore was previously imported outside its own
# try/except and was missing from requirements.txt, so a clean install crashed
# on startup with ModuleNotFoundError before anything else ran.
from backend.net import apply_ssl_workarounds

apply_ssl_workarounds()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from backend.board_builder import build_board, launch_board
from backend.scraper import fetch_all_candidates
from backend.slots import profile_for
from backend.taxonomy import TaxonomyEngine
from backend.validator import BUILD, ReferenceValidator, select_board

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Assemblage")
app.mount("/static", StaticFiles(directory=os.path.join(APP_ROOT, "static")), name="static")

EXPORT_DIR = os.path.join(tempfile.gettempdir(), "assemblage_boards")
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "assemblage_previews")
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")

validator = ReferenceValidator()
taxonomy_engine = TaxonomyEngine(validator.model, validator.tokenizer, validator.device)


class PromptRequest(BaseModel):
    prompt: str
    target_count: int = 12


def _safe_join(base: str, *parts: str) -> str:
    """
    Joins user-supplied path components, rejecting traversal.

    The download and preview routes previously passed request path segments
    straight into os.path.join, so '..%2F..%2F' style input could read files
    outside the export directory.
    """
    for part in parts:
        if not SAFE_NAME.match(part):
            raise HTTPException(status_code=400, detail="Invalid path.")
    path = os.path.abspath(os.path.join(base, *parts))
    if not path.startswith(os.path.abspath(base) + os.sep):
        raise HTTPException(status_code=400, detail="Invalid path.")
    return path


def save_thumbnail(img: Image.Image, save_path: str, max_dim: int = 400) -> None:
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    thumb.save(save_path, format="JPEG", quality=80, optimize=True)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(APP_ROOT, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health():
    """Used by the Blender add-on to poll for server readiness."""
    return {"status": "ok", "device": validator.device, "build": BUILD}


@app.post("/api/generate")
async def generate(req: PromptRequest):
    async def run_pipeline():
        session_id = uuid.uuid4().hex[:8]
        session_preview_dir = os.path.join(PREVIEW_DIR, session_id)
        os.makedirs(session_preview_dir, exist_ok=True)
        loop = asyncio.get_event_loop()

        # 1. Slot-tagged query expansion
        yield _sse({"status": "Reading the subject", "stage": 1})
        slot_queries = await taxonomy_engine.generate_reference_queries(req.prompt)
        yield _sse({
            "status": "Built four searches",
            "stage": 2,
            "queries": [{"slot": sq.slot, "label": profile_for(sq.slot)["label"],
                         "query": sq.query} for sq in slot_queries],
        })

        # 2. Ingestion, preserving which slot found each image
        yield _sse({"status": "Searching", "stage": 3})
        candidates = await fetch_all_candidates(slot_queries)
        if not candidates:
            yield _sse({"error": "Nothing downloaded. Check your connection, "
                                 "or try a shorter subject.", "stage": 3})
            return

        yield _sse({"status": f"Found {len(candidates)} images", "stage": 4})

        # 3. Cheap gates and duplicate removal
        survivors = await loop.run_in_executor(
            None, lambda: validator.prefilter_all(candidates)
        )
        if not survivors:
            yield _sse({"error": "Everything failed the sharpness and size checks. "
                                 "Try a more common subject.", "stage": 4})
            return

        # 4. Slot-aware semantic scoring
        yield _sse({"status": f"Sorting {len(survivors)} into slots", "stage": 5})
        scored = await loop.run_in_executor(
            None, lambda: validator.score_candidates(survivors, req.prompt)
        )
        if not scored:
            yield _sse({"error": "Nothing matched the subject closely enough. "
                                 "Try naming the object more directly.", "stage": 5})
            return

        # 5. Per-slot quota selection
        board = select_board(scored, req.target_count)

        breakdown: dict = {}
        for item in board:
            label = profile_for(item.slot)["label"]
            breakdown[label] = breakdown.get(label, 0) + 1
        yield _sse({"status": f"Packing {len(board)} images", "stage": 6,
                    "breakdown": breakdown})

        # 6. Previews. Each carries its slot and source so the client can group
        # results by category rather than showing an undifferentiated grid.
        previews = []
        for idx, item in enumerate(board):
            thumb_filename = f"thumb_{idx}.jpg"
            save_thumbnail(item.image, os.path.join(session_preview_dir, thumb_filename))
            previews.append({
                "url": f"/api/preview/{session_id}/{thumb_filename}",
                "slot": item.slot,
                "label": profile_for(item.slot)["label"],
                "source": item.url,
                "score": round(item.score, 4),
                "subject": round(item.subject, 4),
            })

        # 7. Compile the .pur canvas
        slug = re.sub(r"[^A-Za-z0-9]+", "_", req.prompt)[:24].strip("_") or "board"
        filename = f"assemblage_{slug}_{session_id}.pur"
        out_path = os.path.join(EXPORT_DIR, filename)

        entries = [
            (item.image, profile_for(item.slot)["label"], item.url) for item in board
        ]
        await loop.run_in_executor(None, lambda: build_board(entries, out_path))
        await loop.run_in_executor(None, lambda: launch_board(out_path))

        yield _sse({
            "complete": True,
            "count": len(board),
            "breakdown": breakdown,
            "download_url": f"/api/download/{filename}",
            "previews": previews,
        })

    return StreamingResponse(run_pipeline(), media_type="text/event-stream")


@app.get("/api/preview/{session_id}/{filename}")
async def get_preview(session_id: str, filename: str):
    path = _safe_join(PREVIEW_DIR, session_id, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Image not found.")


@app.get("/api/download/{filename}")
async def download(filename: str):
    path = _safe_join(EXPORT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename, media_type="application/octet-stream")
    raise HTTPException(status_code=404, detail="File not found.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)