from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api import health_router, ingest_router, search_router
from .config import settings

app = FastAPI(
    title="Local Multimodal RAG API",
    description="Offline-first Multimodal RAG powered by Google EmbeddingGemma-2 & LanceDB",
    version="1.0.0"
)

# Enable CORS for local development and future Vercel frontend deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(search_router)

# Mount Static Uploads (for browser preview of images, audio, video frames)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOADS_DIR)), name="uploads")

# Mount Static Assets & Dashboard
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def root():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Local Multimodal RAG Server is ready.",
        "docs": "/docs"
    }
