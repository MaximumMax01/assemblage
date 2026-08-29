import os
import io
import json
import uuid
import tempfile
import asyncio
import sys

# Ensure backend directory is in Python search path
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Link macOS Keychain for school/enterprise SSL networks
import truststore
try:
    os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
    os.environ["CURL_CA_BUNDLE"] = ""
    truststore.inject_into_ssl()
except ImportError:
    os.environ["PYTHONHTTPSVERIFY"] = "0"
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

from backend.taxonomy import TaxonomyEngine
from backend.scraper import fetch_all_candidates
from backend.validator import ReferenceValidator
from backend.pur_writer import build_and_launch_pur

app = FastAPI(title="ARCA MVP")
app.mount("/static", StaticFiles(directory="static"), name="static")

EXPORT_DIR = os.path.join(tempfile.gettempdir(), "arca_boards")
PREVIEW_DIR = os.path.join(tempfile.gettempdir(), "arca_previews")
os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(PREVIEW_DIR, exist_ok=True)

# Initialize models
validator = ReferenceValidator()
taxonomy_engine = TaxonomyEngine(validator.model, validator.tokenizer, validator.device)

class PromptRequest(BaseModel):
    prompt: str
    target_count: int = 12

def save_thumbnail(img: Image.Image, save_path: str, max_dim: int = 400) -> None:
    """Converts and saves a clean JPEG thumbnail to the preview directory."""
    thumb = img.convert("RGB").copy()
    thumb.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    thumb.save(save_path, format="JPEG", quality=80, optimize=True)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/generate")
async def generate(req: PromptRequest):
    async def run_pipeline():
        session_id = str(uuid.uuid4())[:8]
        session_preview_dir = os.path.join(PREVIEW_DIR, session_id)
        os.makedirs(session_preview_dir, exist_ok=True)
        
        # 1. Vector Taxonomy Expansion
        yield f"data: {json.dumps({'status': 'Analyzing prompt with OpenCLIP embeddings...'})}\n\n"
        queries = await taxonomy_engine.generate_reference_queries(req.prompt)
        yield f"data: {json.dumps({'status': 'Generated specialized queries', 'queries': queries})}\n\n"

        # 2. Ingestion
        yield f"data: {json.dumps({'status': 'Fetching high-resolution image candidates...'})}\n\n"
        raw_candidates = await fetch_all_candidates(queries)
        yield f"data: {json.dumps({'status': f'Downloaded {len(raw_candidates)} payloads. Running OpenCV & OpenCLIP filters...'})}\n\n"

        # 3. Vision & Semantic Quality Verification
        loop = asyncio.get_event_loop()
        def _validate_all():
            scored = []
            for img_bytes, _ in raw_candidates:
                valid, img, score = validator.validate_and_score(img_bytes, req.prompt)
                if valid and img:
                    scored.append((score, img))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [img for _, img in scored[:req.target_count]]

        verified_images = await loop.run_in_executor(None, _validate_all)

        if not verified_images:
            yield f"data: {json.dumps({'error': 'No reference images passed the quality filter. Try refining the prompt.'})}\n\n"
            return

        yield f"data: {json.dumps({'status': f'Verified {len(verified_images)} assets. Assembling preview & PureRef canvas...'})}\n\n"

        # Save thumbnails as clean static preview files
        preview_urls = []
        for idx, img in enumerate(verified_images):
            thumb_filename = f"thumb_{idx}.jpg"
            thumb_path = os.path.join(session_preview_dir, thumb_filename)
            save_thumbnail(img, thumb_path)
            preview_urls.append(f"/api/preview/{session_id}/{thumb_filename}")

        # 4. PureRef Compilation & Launch
        filename = f"ARCA_{req.prompt[:16].strip().replace(' ', '_')}_{session_id}.pur"
        out_path = os.path.join(EXPORT_DIR, filename)
        await loop.run_in_executor(None, lambda: build_and_launch_pur(verified_images, out_path))

        yield f"data: {json.dumps({'complete': True, 'count': len(verified_images), 'download_url': f'/api/download/{filename}', 'previews': preview_urls})}\n\n"

    return StreamingResponse(run_pipeline(), media_type="text/event-stream")

@app.get("/api/preview/{session_id}/{filename}")
async def get_preview(session_id: str, filename: str):
    path = os.path.join(PREVIEW_DIR, session_id, filename)
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    return {"error": "Image not found"}

@app.get("/api/download/{filename}")
async def download(filename: str):
    path = os.path.join(EXPORT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename, media_type="application/octet-stream")
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)