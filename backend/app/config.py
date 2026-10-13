import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    # Application Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent.parent / "data")))
    LANCEDB_DIR: Path = Path(os.getenv("LANCEDB_DIR", str(DATA_DIR / "lancedb")))
    UPLOADS_DIR: Path = Path(os.getenv("UPLOADS_DIR", str(DATA_DIR / "uploads")))
    CACHE_DIR: Path = Path(os.getenv("CACHE_DIR", str(DATA_DIR / "cache")))
    TABLE_NAME: str = os.getenv("TABLE_NAME", "knowledge_base")

    # Embedding Model Settings (google/embeddinggemma-2)
    EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "google/embeddinggemma-2")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 256))
    NORMALIZE_EMBEDDINGS: bool = True
    USE_BFLOAT16: bool = True

    # Task Instruction Prefixes (Google Best Practice)
    QUERY_PROMPT_NAME: str = "SearchQuery"
    DOC_PROMPT_NAME: str = "Document"

    # Modality Sampling Settings
    AUDIO_SAMPLE_RATE: int = 16000            # Mono 16 kHz
    VIDEO_FPS: int = 1                        # 1 frame per second
    MAX_CONTEXT_TOKENS: int = 8192

    # LLM Settings (In-Process Transformers / HuggingFace)
    LLM_MODEL: str = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

    # API Settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

settings = Settings()
