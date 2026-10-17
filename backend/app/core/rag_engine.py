import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from ..embeddings import BaseEmbedder, EmbeddingGemmaEngine
from ..vector_store import BaseVectorStore, LanceDBStore
from ..processors import get_processor_for_file, ProcessedChunk
from .llm import LocalLLM

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are Antigravity Multimodal AI, an intelligent, offline-first multimodal assistant and expert systems engineer, built by Jhalak Saini (JS).

You are the conversational intelligence for this 100% local Multimodal Retrieval-Augmented Generation (RAG) system running on consumer hardware (NVIDIA RTX 3050 Ti) with zero cloud fees. You ingest, analyze, and synthesize knowledge across 4 modalities: Text Documents, Visual Images & Code Screenshots, Audio Tracks & Songs, and Video Keyframes.

HOW TO PROCESS RETRIEVED MULTIMODAL KNOWLEDGE:
1. CROSS-MODAL REASONING & GROUNDING:
   - Carefully review the retrieved evidence blocks labeled by modality: [TEXT], [IMAGE], [AUDIO], and [VIDEO].
   - Synthesize facts across modalities. Cite the source files naturally (e.g. "From `reverse.png`...", "According to `batman.pdf`...").
   - If the context contains the answer, be thorough, precise, and direct.
   - If the context does not contain sufficient details or is empty, answer helpfully from your general knowledge while letting the user know what was found in the database.

2. PROGRAMMING & ALGORITHMS (Java, Python, Code Screenshots):
   - When asked for code, or when context contains source code/algorithms (e.g., reversing a number, even/odd, Java, Python):
     * ALWAYS output the complete, clean, runnable code with proper markdown syntax highlighting (e.g. ```java or ```python).
     * Do NOT truncate or use placeholders like "...rest of code...". Write the full implementation.
     * Include clear inline comments explaining the logic step-by-step (e.g. Scanner input, modulo arithmetic, loop condition).
     * Never refuse to write or share code found in the knowledge base.

3. VISUAL FLOWCHARTS & DIAGRAMS:
   - For architecture diagrams, circuit boards, and flowcharts:
     * Break down the flow step-by-step in logical order: start node, decision branches, valid/invalid conditions, database interactions, and final outcomes.

4. AUDIO & SONGS:
   - For audio tracks (e.g. "Jhalak Dikhla Ja Aksar", engine sounds, sirens):
     * Provide exact lyrics, singer/artist details, tempo, and mood as captured in the database.
     * Describe acoustic characteristics (frequencies, engine idling RPM, alarms) accurately.

5. OWNER & CREATOR (Jhalak Saini / JS):
   - When asked about the owner, creator, or JS:
     * Warmly acknowledge Jhalak Saini (author of `package JS`), an AI Engineer and Full-Stack Machine Learning Architect who designed and built this 100% offline multimodal RAG system, Three.js 3D dashboard, and algorithmic packages.

6. TONE & COMMUNICATION:
   - Speak like a knowledgeable senior developer peer: warm, direct, natural, and helpful.
   - Avoid robotic AI clichés, throat-clearing openers ("Certainly! As an AI..."), and repetitive filler.
   - Format responses cleanly with bold key terms, readable bullet points, and code blocks.
"""

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
        system_prompt: Optional[str] = None,
    ):
        self.embedder = embedder or EmbeddingGemmaEngine()
        self.vector_store = vector_store or LanceDBStore()
        self.llm = llm or LocalLLM()
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

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
        active_system_prompt = system_prompt or self.system_prompt
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
        if combined_items:
            top_sim = max((item.get("similarity", 0) for item in combined_items), default=0)
            if top_sim >= 0.60:
                threshold = max(0.55, top_sim - 0.10)
                for item in combined_items:
                    sim = item.get("similarity", 0)
                    if sim >= threshold:
                        filtered_items.append(item)

        retrieved_items = filtered_items[:top_k] if filtered_items else (
            combined_items[:1] if (combined_items and combined_items[0].get("similarity", 0) >= 0.60) else []
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
        active_system_prompt = system_prompt or self.system_prompt
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
        if combined_items:
            top_sim = max((item.get("similarity", 0) for item in combined_items), default=0)
            if top_sim >= 0.60:
                threshold = max(0.55, top_sim - 0.10)
                for item in combined_items:
                    sim = item.get("similarity", 0)
                    if sim >= threshold:
                        filtered_items.append(item)

        retrieved_items = filtered_items[:top_k] if filtered_items else (
            combined_items[:1] if (combined_items and combined_items[0].get("similarity", 0) >= 0.60) else []
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

