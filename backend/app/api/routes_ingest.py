import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from ..config import settings
from ..core import MultimodalRAGEngine
from ..processors import PROCESSOR_REGISTRY

router = APIRouter(prefix="/ingest", tags=["Ingest"])
engine = MultimodalRAGEngine()

@router.get("/list")
async def list_files():
    """Lists all files stored in data/uploads/ with their size and format."""
    try:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        supported = set(PROCESSOR_REGISTRY.keys())
        files = []
        for f in settings.UPLOADS_DIR.iterdir():
            if f.is_file() and f.suffix.lower() in supported and not f.name.endswith("_16k_mono.wav"):
                mod = "document"
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
                    mod = "image"
                elif f.suffix.lower() in [".wav", ".mp3", ".ogg", ".flac"]:
                    mod = "audio"
                elif f.suffix.lower() in [".mp4", ".mkv", ".mov"]:
                    mod = "video"

                size_kb = round(f.stat().st_size / 1024, 1)
                size_str = f"{size_kb} KB" if size_kb < 1024 else f"{round(size_kb / 1024, 2)} MB"
                files.append({
                    "name": f.name,
                    "extension": f.suffix.lower(),
                    "modality": mod,
                    "size": size_str,
                    "url": f"/uploads/{f.name}"
                })

        files.sort(key=lambda x: x["name"])
        return {"files": files, "total": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/file")
async def ingest_file(file: UploadFile = File(...)):
    """Uploads and ingests any supported media or document file."""
    try:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = settings.UPLOADS_DIR / file.filename
        with dest_path.open("wb") as buffer:
            buffer.write(await file.read())

        chunks_created = engine.ingest_file(dest_path)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": chunks_created
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/scan")
async def scan_uploads_folder():
    """Scans data/uploads/ and indexes any new or existing files."""
    try:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        supported = set(PROCESSOR_REGISTRY.keys())
        files = [
            f for f in settings.UPLOADS_DIR.iterdir()
            if f.is_file() and f.suffix.lower() in supported and not f.name.endswith("_16k_mono.wav")
        ]

        indexed_summary = []
        for f in files:
            try:
                chunks = engine.ingest_file(f)
                indexed_summary.append({"filename": f.name, "chunks": chunks, "status": "success"})
            except Exception as e:
                indexed_summary.append({"filename": f.name, "error": str(e), "status": "failed"})

        return {
            "status": "completed",
            "files_scanned": len(files),
            "details": indexed_summary,
            "total_documents_in_db": engine.vector_store.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
