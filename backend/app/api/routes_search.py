from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from ..core import MultimodalRAGEngine

router = APIRouter(prefix="/query", tags=["Query & Search"])
engine = MultimodalRAGEngine()

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    modality: Optional[str] = None  # None for all, or "text", "image", "audio", "video"

@router.post("/search")
async def similarity_search(req: QueryRequest):
    """Vector similarity search across modalities (no LLM generation)."""
    try:
        results = engine.retrieve(req.query, top_k=req.top_k, modality=req.modality)
        return {"query": req.query, "modality": req.modality, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask")
async def rag_ask(req: QueryRequest):
    """Full Multimodal RAG: cross-modal retrieval + local Qwen2.5 conversational answer generation."""
    try:
        result = await engine.answer(req.query, top_k=req.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
