import logging
from typing import Any, List, Optional
import numpy as np
from .base import BaseEmbedder
from ..config import settings

logger = logging.getLogger(__name__)

class EmbeddingGemmaEngine(BaseEmbedder):
    """
    Embedding engine for google/embeddinggemma-2.
    Implements Google DeepMind's best practices:
      - Uses bfloat16 numerical precision (never float16).
      - Applies Matryoshka dimension truncation (256d) with L2 normalization.
      - Uses asymmetric task prefixes: 'SearchQuery' for queries, 'Document' for texts.
      - Supports multimodal inputs and selective encoder loading.
    """

    def __init__(
        self,
        model_id: str = settings.EMBEDDING_MODEL_ID,
        truncate_dim: int = settings.EMBEDDING_DIM,
        normalize: bool = settings.NORMALIZE_EMBEDDINGS,
        device: str = "cuda"
    ):
        self.model_id = model_id
        self.truncate_dim = truncate_dim
        self.normalize = normalize
        self.device = device
        self.model = None

    def load_model(self, active_modalities: Optional[List[str]] = None) -> None:
        """
        Loads the model using sentence-transformers.
        If active_modalities is provided, disables unused encoders to conserve VRAM.
        """
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers and torch are not yet installed.")

        # Ensure correct bfloat16 / float32 precision
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
        logger.info(f"Loading {self.model_id} on {self.device} with dtype={dtype}")

        # Selective encoder configuration if needed
        config_kwargs = {}
        if active_modalities is not None:
            if "vision" not in active_modalities and "image" not in active_modalities:
                config_kwargs["vision_config"] = None
            if "audio" not in active_modalities:
                config_kwargs["audio_config"] = None

        self.model = SentenceTransformer(
            self.model_id,
            device=self.device,
            model_kwargs={"torch_dtype": dtype},
            config_kwargs=config_kwargs if config_kwargs else None
        )
        logger.info("EmbeddingGemma-2 model loaded successfully.")

    def embed_query(self, query: str) -> np.ndarray:
        """Encodes search query with SearchQuery prompt name."""
        if self.model is None:
            self.load_model()

        return self.model.encode(
            query,
            prompt_name=settings.QUERY_PROMPT_NAME,
            truncate_dim=self.truncate_dim,
            normalize_embeddings=self.normalize,
        )

    def embed_corpus(self, items: List[Any], modality: str = "text") -> np.ndarray:
        """
        Encodes corpus items.
        For text: if not already formatted with title, applies Document prompt.
        For media (images, audio, video): passed raw without text prompt.
        """
        if self.model is None:
            self.load_model()

        # Text requires prompt_name or manual formatting; media should NOT have text prompt
        prompt_name = settings.DOC_PROMPT_NAME if modality == "text" else None

        return self.model.encode(
            items,
            prompt_name=prompt_name,
            truncate_dim=self.truncate_dim,
            normalize_embeddings=self.normalize,
            batch_size=16
        )
