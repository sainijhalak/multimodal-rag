from abc import ABC, abstractmethod
from typing import Any, List, Union
import numpy as np

class BaseEmbedder(ABC):
    """Abstract interface for embedding engines."""

    @abstractmethod
    def load_model(self) -> None:
        """Loads model into memory/GPU."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single user search query."""
        pass

    @abstractmethod
    def embed_corpus(self, items: List[Any], modality: str = "text") -> np.ndarray:
        """Embeds a batch of corpus items (text, image paths, audio paths, or interleaved dicts)."""
        pass
