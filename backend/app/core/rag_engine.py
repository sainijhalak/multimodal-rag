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

        modalities = [c.modality for c in chunks]
        metadatas = [c.metadata for c in chunks]
        modality = modalities[0]

        # Prepare items for embedder based on modality
        if modality in ("image", "video"):
            # Load images / keyframes via PIL for vision encoder
            embed_items = [Image.open(c.content) for c in chunks]
        elif modality == "audio":
            # Audio paths passed directly
            embed_items = [str(c.content) for c in chunks]
        else:
            # Text content
            embed_items = [str(c.content) for c in chunks]

        # Vectorize
        vectors = self.embedder.embed_corpus(embed_items, modality=modality)

        # Content strings stored in DB
        db_contents = [str(c.content) for c in chunks]
        self.vector_store.add(vectors, db_contents, modalities, metadatas)
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 5, modality: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves top_k nearest multimodal items using SearchQuery prefix."""
        query_vector = self.embedder.embed_query(query)
        results = self.vector_store.search(query_vector, top_k=top_k, modality=modality)
        return results

    async def answer(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Performs RAG: retrieves relevant chunks and asks the local LLM."""
        retrieved_items = self.retrieve(query, top_k=top_k)

        # Format multimodal context with source types and timestamps
        context_parts = []
        for rank, item in enumerate(retrieved_items, start=1):
            mod = item.get("modality", "text").upper()
            src = item.get("source", "unknown")
            title = item.get("title", "")
            cnt = item.get("content", "")

            if mod == "TEXT":
                context_parts.append(f"[{rank}] {mod} Document '{title}' (from {src}):\n{cnt}")
            elif mod == "VIDEO":
                context_parts.append(f"[{rank}] {mod} Keyframe '{title}' (from {src})")
            elif mod == "IMAGE":
                context_parts.append(f"[{rank}] {mod} Image '{title}' (from {src})")
            elif mod == "AUDIO":
                context_parts.append(f"[{rank}] {mod} Audio Clip '{title}' (from {src})")

        context_str = "\n\n".join(context_parts)

        system_prompt = (
            "You are a helpful, accurate multimodal AI assistant. "
            "Answer the user's question clearly and concisely based on the retrieved context below. "
            "Cite relevant documents, images, audio, or video timestamps when answering."
        )

        user_prompt = f"Retrieved Multimodal Knowledge:\n{context_str}\n\nUser Question: {query}\n\nAnswer:"
        response = await self.llm.generate_response(user_prompt, system_prompt=system_prompt)

        return {
            "query": query,
            "answer": response,
            "sources": retrieved_items
        }
