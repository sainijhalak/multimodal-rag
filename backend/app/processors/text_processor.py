import logging
from pathlib import Path
from typing import List
from .base import BaseProcessor, ProcessedChunk

logger = logging.getLogger(__name__)

class TextProcessor(BaseProcessor):
    """Handles text documents (.txt, .md, .pdf, .json, .csv)."""

    @property
    def supported_modality(self) -> str:
        return "text"

    def process(self, file_path: Path, chunk_size: int = 500, chunk_overlap: int = 50) -> List[ProcessedChunk]:
        """Reads a text or PDF file, splits it into manageable chunks with title metadata."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # PDF Extraction
        if file_path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(str(file_path))
                text = "\n".join([page.extract_text() or "" for page in reader.pages])
            except Exception as e:
                logger.error(f"Error reading PDF {file_path}: {e}")
                text = file_path.read_text(encoding="utf-8", errors="ignore")
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")

        title = file_path.stem.replace("_", " ").title()

        # Word-based chunking with overlap
        words = text.split()
        if not words:
            return []

        chunks = []
        i = 0
        step = max(1, chunk_size - chunk_overlap)
        while i < len(words):
            chunk_text = " ".join(words[i:i + chunk_size])
            formatted_text = f"title: {title} | text: {chunk_text}"
            chunks.append(
                ProcessedChunk(
                    content=formatted_text,
                    modality="text",
                    metadata={
                        "source": str(file_path.name),
                        "title": title,
                        "chunk_index": len(chunks),
                        "modality": "text"
                    }
                )
            )
            i += step

        return chunks
