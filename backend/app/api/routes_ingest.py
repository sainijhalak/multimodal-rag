from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from ..config import settings
from ..core import MultimodalRAGEngine

router = APIRouter(prefix="/ingest", tags=["Ingest"])
engine = MultimodalRAGEngine()

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
