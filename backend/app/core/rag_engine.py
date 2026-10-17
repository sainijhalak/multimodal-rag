import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from ..embeddings import BaseEmbedder, EmbeddingGemmaEngine
from ..vector_store import BaseVectorStore, LanceDBStore
from ..processors import get_processor_for_file, ProcessedChunk
from .llm import LocalLLM

logger = logging.getLogger(__name__)

class MultimodalRAGEngine:
    """
    Central coordinator orchestrating:
      1. Ingestion via modular Processors (Text, Image, Audio, Video)
      2. Vectorization via EmbeddingGemmaEngine (MRL 256d, bfloat16, SearchQuery/Document)
      3. Indexing & Storage via LanceDB
      4. Grounded retrieval & Answer generation via LocalLLM (Qwen2.5-0.5B-Instruct)
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        llm: Optional[LocalLLM] = None,
    ):
        self.embedder = embedder or EmbeddingGemmaEngine()
        self.vector_store = vector_store or LanceDBStore()
        self.llm = llm or LocalLLM()

    def ingest_file(self, file_path: Path) -> int:
        """Processes a file with its registered processor and stores vectors."""
        file_path = Path(file_path)
        processor = get_processor_for_file(file_path.suffix)
        chunks: List[ProcessedChunk] = processor.process(file_path)

        if not chunks:
            return 0

        # Group chunks by modality so mixed outputs (e.g. image + OCR text) are encoded properly
        from collections import defaultdict
        chunks_by_mod = defaultdict(list)
        for c in chunks:
            chunks_by_mod[c.modality].append(c)

        for mod, mod_chunks in chunks_by_mod.items():
            if mod in ("image", "video"):
                embed_items = [Image.open(c.content) for c in mod_chunks]
            elif mod == "audio":
                embed_items = [str(c.content) for c in mod_chunks]
            else:
                embed_items = [str(c.content) for c in mod_chunks]

            vectors = self.embedder.embed_corpus(embed_items, modality=mod)
            db_contents = [str(c.content) for c in mod_chunks]
            modalities = [c.modality for c in mod_chunks]
            metadatas = [c.metadata for c in mod_chunks]
            self.vector_store.add(vectors, db_contents, modalities, metadatas)

        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5, modality: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves top_k nearest multimodal items using SearchQuery prefix."""
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k, modality=modality)
        return results

    async def answer(
        self,
        query: str,
        top_k: int = 4,
        context_file: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Performs RAG: retrieves relevant chunks and asks the local LLM."""
        from .prompts import DEFAULT_SYSTEM_PROMPT, is_conversational_query
        active_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        is_greeting = is_conversational_query(query)

        retrieved_items = []

        # If a specific file was uploaded/attached, fetch its chunks directly
        if context_file:
            try:
                file_items = self.vector_store.search_by_source(context_file)
                retrieved_items.extend(file_items)
            except Exception as e:
                logger.warning(f"Could not fetch chunks for context_file '{context_file}': {e}")

        # Retrieve semantically nearest items
        semantic_items = self.retrieve(query, top_k=top_k)

        # Deduplicate while preserving priority
        seen = set()
        combined_items = []
        for item in (retrieved_items + semantic_items):
            key = (item.get("source"), item.get("modality"), item.get("title"))
            if key not in seen:
                seen.add(key)
                combined_items.append(item)

        # Relevance filtering: retain top-tier matching results while eliminating distant noise
        filtered_items = []
        if combined_items and not is_greeting:
            top_sim = max((item.get("similarity", 0) for item in combined_items), default=0)
            if top_sim >= 0.60:
                threshold = max(0.55, top_sim - 0.10)
                for item in combined_items:
                    sim = item.get("similarity", 0)
                    if sim >= threshold:
                        filtered_items.append(item)

        retrieved_items = filtered_items[:top_k] if filtered_items else (
            [] if is_greeting else (
                combined_items[:1] if (combined_items and combined_items[0].get("similarity", 0) >= 0.60) else []
            )
        )

        # Format multimodal context with source types and timestamps
        if retrieved_items:
            context_parts = []
            for rank, item in enumerate(retrieved_items, start=1):
                mod = item.get("modality", "text").upper()
                src = item.get("source", "unknown")
                title = item.get("title", "")
                cnt = item.get("content", "")

                if mod == "TEXT":
                    context_parts.append(f"[{rank}] {mod} Content '{title}' (from {src}):\n{cnt}")
                elif mod == "VIDEO":
                    context_parts.append(f"[{rank}] {mod} Keyframe '{title}' (from {src})")
                elif mod == "IMAGE":
                    context_parts.append(f"[{rank}] {mod} Image '{title}' (from {src})")
                elif mod == "AUDIO":
                    context_parts.append(f"[{rank}] {mod} Audio Track '{title}' (from {src})")

            context_str = "\n\n".join(context_parts)
            user_prompt = f"Retrieved Multimodal Knowledge:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"
        else:
            user_prompt = f"User Question: {query}\n\nAnswer:"

        response = await self.llm.generate_response(user_prompt, system_prompt=active_system_prompt)

        return {
            "query": query,
            "answer": response,
            "sources": retrieved_items
        }

    async def answer_stream(
        self,
        query: str,
        top_k: int = 4,
        context_file: Optional[str] = None,
        system_prompt: Optional[str] = None
    ):
        """Streaming version of answer(): yields sources immediately, then streams generated tokens."""
        from .prompts import DEFAULT_SYSTEM_PROMPT, is_conversational_query
        active_system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        is_greeting = is_conversational_query(query)

        retrieved_items = []

        if context_file:
            try:
                file_items = self.vector_store.search_by_source(context_file)
                retrieved_items.extend(file_items)
            except Exception as e:
                logger.warning(f"Could not fetch chunks for context_file '{context_file}': {e}")

        # Semantic retrieval in worker thread to prevent event loop blocking
        import asyncio
        semantic_items = await asyncio.to_thread(self.retrieve, query, top_k=top_k)

        # Deduplicate
        seen = set()
        combined_items = []
        for item in (retrieved_items + semantic_items):
            key = (item.get("source"), item.get("modality"), item.get("title"))
            if key not in seen:
                seen.add(key)
                combined_items.append(item)

        # Relevance filtering: retain top-tier matching results while eliminating distant noise
        filtered_items = []
        if combined_items and not is_greeting:
            top_sim = max((item.get("similarity", 0) for item in combined_items), default=0)
            if top_sim >= 0.60:
                threshold = max(0.55, top_sim - 0.10)
                for item in combined_items:
                    sim = item.get("similarity", 0)
                    if sim >= threshold:
                        filtered_items.append(item)

        retrieved_items = filtered_items[:top_k] if filtered_items else (
            [] if is_greeting else (
                combined_items[:1] if (combined_items and combined_items[0].get("similarity", 0) >= 0.60) else []
            )
        )

        # Format context
        if retrieved_items:
            context_parts = []
            for rank, item in enumerate(retrieved_items, start=1):
                mod = item.get("modality", "text").upper()
                src = item.get("source", "unknown")
                title = item.get("title", "")
                cnt = item.get("content", "")

                if mod == "TEXT":
                    context_parts.append(f"[{rank}] {mod} Content '{title}' (from {src}):\n{cnt}")
                elif mod == "VIDEO":
                    context_parts.append(f"[{rank}] {mod} Keyframe '{title}' (from {src})")
                elif mod == "IMAGE":
                    context_parts.append(f"[{rank}] {mod} Image '{title}' (from {src})")
                elif mod == "AUDIO":
                    context_parts.append(f"[{rank}] {mod} Audio Track '{title}' (from {src})")

            context_str = "\n\n".join(context_parts)
            user_prompt = f"Retrieved Multimodal Knowledge:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"
        else:
            user_prompt = f"User Question: {query}\n\nAnswer:"

        # 1. Yield sources immediately so UI renders media cards instantly
        yield {
            "type": "sources",
            "sources": retrieved_items
        }

        # 2. Stream tokens in real time
        async for token in self.llm.stream_response(user_prompt, system_prompt=active_system_prompt):
            yield {
                "type": "token",
                "token": token
            }

        # 3. Final completion signal
        yield {
            "type": "done"
        }

