from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path

class ProcessedChunk:
    """Represents a standardized chunk extracted from any modality."""
    def __init__(
        self,
        content: Any,
        modality: str,
        metadata: Dict[str, Any] = None
    ):
        self.content = content          # Text string, image path, audio path, or interleaved dict
        self.modality = modality        # 'text', 'image', 'audio', 'video', 'interleaved'
        self.metadata = metadata or {}

class BaseProcessor(ABC):
    """Abstract interface for modality-specific ingestion processors."""

    @property
    @abstractmethod
    def supported_modality(self) -> str:
        """Name of the modality handled by this processor (e.g. 'text', 'image')."""
        pass

    @abstractmethod
    def process(self, file_path: Path, **kwargs) -> List[ProcessedChunk]:
        """Reads raw file and returns a list of processed chunks ready for embedding."""
        pass
