from fastapi import APIRouter
import torch
from ..config import settings
from ..vector_store import LanceDBStore

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("/")
async def health_check():
    cuda_available = torch.cuda.is_available()
    vram_mb = 0.0
    vram_reserved = 0.0
    device_name = "CPU"

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        vram_mb = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
        vram_reserved = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 1)

    doc_count = 0
    try:
        store = LanceDBStore()
        doc_count = store.count()
    except Exception:
        pass

    return {
        "status": "healthy",
        "system": "100% Local Multimodal RAG Engine",
        "device": device_name,
        "cuda_available": cuda_available,
        "vram_allocated_mb": vram_mb,
        "vram_reserved_mb": vram_reserved,
        "total_documents_indexed": doc_count,
        "embedding_model": settings.EMBEDDING_MODEL_ID,
        "llm_model": settings.LLM_MODEL,
        "modalities_supported": ["text", "image", "audio", "video"],
        "cost": "$0.00 (Zero Cloud Fees)"
    }
