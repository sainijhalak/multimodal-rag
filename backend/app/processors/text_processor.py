from pathlib import Path
from typing import List
from .base import BaseProcessor, ProcessedChunk

class TextProcessor(BaseProcessor):
    """Handles text documents, markdown, and plain files."""

    @property
    def supported_modality(self) -> str:
        return "text"

    def process(self, file_path: Path, chunk_size: int = 500, chunk_overlap: int = 50) -> List[ProcessedChunk]:
        """Reads a text file, splits it into manageable chunks with title metadata."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        title = file_path.stem.replace("_", " ").title()

        # Simple character/word based chunking for text
        words = text.split()
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
                    }
                )
            )
            i += step

        return chunks
