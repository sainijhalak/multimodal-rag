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
    context_file: Optional[str] = None  # Optional filename of attached/active file
    system_prompt: Optional[str] = None  # Optional custom system prompt override

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
        result = await engine.answer(
            req.query,
            top_k=req.top_k,
            context_file=req.context_file,
            system_prompt=req.system_prompt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask_stream")
async def rag_ask_stream(req: QueryRequest):
    """Real-time Streaming Multimodal RAG: SSE stream with instant sources and token-by-token text."""
    from fastapi.responses import StreamingResponse
    import json

    async def event_generator():
        try:
            async for chunk in engine.answer_stream(
                req.query,
                top_k=req.top_k,
                context_file=req.context_file,
                system_prompt=req.system_prompt
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

