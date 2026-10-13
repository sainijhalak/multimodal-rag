import sys
from pathlib import Path
import uvicorn

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(backend_dir))

from app.config import settings

if __name__ == "__main__":
    print("=" * 65)
    print("🚀 Starting Multimodal RAG Server on http://localhost:8000")
    print("🎓 Models: Google EmbeddingGemma-2 + Qwen2.5-0.5B-Instruct")
    print("=" * 65)
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False
    )
