from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np

class BaseVectorStore(ABC):
    """Abstract interface for local vector storage."""

    @abstractmethod
    def initialize(self) -> None:
        """Connects to the database and ensures tables/collections exist."""
        pass

    @abstractmethod
    def add(
        self,
        vectors: np.ndarray,
        contents: List[str],
        modalities: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Adds vectorized items along with raw contents and metadata."""
        pass

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds nearest items matching the query vector."""
        pass
