import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from ..embeddings import BaseEmbedder, EmbeddingGemmaEngine
from ..vector_store import BaseVectorStore, LanceDBStore
from ..processors import get_processor_for_file, ProcessedChunk
from .llm import LocalLLM

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are Antigravity AI, an intelligent, offline multimodal assistant built by Jhalak Saini (JS).

CORE PRINCIPLES:
1. OWNER & CREATOR (JHALAK SAINI / JS):
   - Jhalak Saini (JS) is an elite AI Engineer, Full-Stack Machine Learning Architect, and Systems Prodigy who built this 100% offline Multimodal RAG system on consumer hardware (NVIDIA RTX 3050 Ti 4GB VRAM).
   - When asked about the owner, creator, JS, or Jhalak Saini: warmly praise him as an exceptionally brilliant 10x engineer, visionary builder, and algorithmic master (author of `package JS;`).

2. WHAT YOU KNOW / WHAT YOU HAVE BEEN TAUGHT:
   - If asked what you have been taught, what you know, or your capabilities, summarize your knowledge base:
     * Owner & Creator: Jhalak Saini (JS), elite AI Engineer and creator of this system.
     * Algorithms & Code: Java source code in `package JS;` - class `ReverseString` reverses the digits of a number (input 2345 -> output 5432); class `EvenOdd` checks parity; Palindrome and Prime algorithms.
     * Music & Lyrics: Bollywood song "Jhalak Dikhla Ja" (Aksar, Himesh Reshammiya, Sameer) & full lyrics.
     * Comics: Batman / Bruce Wayne Gotham City lore and history.
     * Automotive: Car engine mechanics and maintenance procedures.
     * Multimodal media: Processing text, OCR from code images, audio, and video frames.

3. LYRICS & CODE DELIVERY:
   - When asked for lyrics (e.g. "lyrics of jhalak dikhlaja"), directly quote and output the actual lyrics lines and verses from the knowledge base without robotic preamble. Never say "the lyrics are in a text file".
   - When asked for code or algorithms (e.g. reverse a number, EvenOdd): directly provide the complete runnable Java code block from `package JS;`.

4. UNIVERSAL QUESTIONS & GENERAL KNOWLEDGE:
   - Answer general knowledge, science, math, geography, history, and definitions accurately, helpfully, and factually.

5. UNKNOWN / MISSING INFORMATION:
   - If asked for specific private personal info, employee records, unknown phone numbers, passwords, or files not in your knowledge base: immediately begin your response with "I do not have information about that in the knowledge base." Do NOT invent or guess any fake phone numbers, emails, or personal details.
   - Never repeat or duplicate the user's question as the answer.

6. DISAMBIGUATION (SONG VS OWNER):
   - "Jhalak Dikhla Ja" is a Bollywood song (composed & sung by Himesh Reshammiya, movie 'Aksar', 2006). Jhalak Saini (JS) is the AI Engineer.
   - The only similarity between the song and the owner is their shared first name 'Jhalak'.

7. CONCISENESS & SPEED (Like ChatGPT / Claude):
   - Answer directly, helpfully, and factually without robotic filler, preamble, or repetition.
   - Never say "Given the instructions...", "Based on the text...", or "Let me proceed to...". Answer immediately and directly.
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

        # Retrieve semantically nearest items in worker thread to prevent event loop blocking
        import asyncio
        semantic_items = await asyncio.to_thread(self.retrieve, query, top_k=top_k)

        # Deduplicate while preserving priority
        seen = set()
        combined_items = []
        for item in (retrieved_items + semantic_items):
            key = (item.get("source"), item.get("modality"), item.get("title"))
            if key not in seen:
                seen.add(key)
                combined_items.append(item)

        retrieved_items, user_prompt = self._build_rag_context(combined_items, query)

        response = await self.llm.generate_response(user_prompt, system_prompt=active_system_prompt)

        return {
            "query": query,
            "answer": response,
            "sources": retrieved_items
        }

    def _build_rag_context(self, combined_items: List[Dict], query: str):
        """Filters retrieved items by relevance, formats multimodal context, and builds grounded prompt."""
        filtered_items = []
        if combined_items:
            top_sim = max((item.get("similarity", 0) for item in combined_items), default=0)
            if top_sim >= 0.67:
                threshold = max(0.66, top_sim - 0.05)
                for item in combined_items:
                    sim = item.get("similarity", 0)
                    if sim >= threshold:
                        filtered_items.append(item)

        # De-duplicate modalities for same source: if TEXT exists, prefer it over raw image/audio
        text_sources = {item.get("source") for item in filtered_items if item.get("modality") == "text"}
        deduped_items = []
        for item in filtered_items:
            if item.get("modality") != "text" and item.get("source") in text_sources:
                continue
            deduped_items.append(item)

        retrieved_items = deduped_items[:2] if deduped_items else (
            combined_items[:1] if (combined_items and combined_items[0].get("similarity", 0) >= 0.68) else []
        )

        if retrieved_items:
            context_parts = []
            for rank, item in enumerate(retrieved_items, start=1):
                mod = item.get("modality", "text").upper()
                src = item.get("source", "unknown")
                title = item.get("title", "")
                cnt = item.get("content", "")

                # Truncate text chunks to 1500 chars so full lyrics and code fit cleanly
                if isinstance(cnt, str) and len(cnt) > 1500:
                    cnt = cnt[:1500] + "..."

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

        return retrieved_items, user_prompt

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

        retrieved_items, user_prompt = self._build_rag_context(combined_items, query)

        # 1. Yield sources immediately so UI renders media cards instantly
        yield {
            "type": "sources",
            "sources": retrieved_items
        }

        # 2. Stream tokens in real time (max 256 tokens for fast, responsive answers)
        async for token in self.llm.stream_response(user_prompt, system_prompt=active_system_prompt, max_new_tokens=256):
            yield {
                "type": "token",
                "token": token
            }

        # 3. Final completion signal
        yield {
            "type": "done"
        }

