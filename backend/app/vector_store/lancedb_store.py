import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from .base import BaseVectorStore
from ..config import settings

logger = logging.getLogger(__name__)

class LanceDBStore(BaseVectorStore):
    """
    LanceDB local vector store.
    Embedded directly in-process with zero background daemon overhead.
    Stores and retrieves multi-modal records across Text, Image, Audio, and Video.
    """

    def __init__(self, db_path: Path = settings.LANCEDB_DIR, table_name: str = settings.TABLE_NAME):
        self.db_path = db_path
        self.table_name = table_name
        self.db = None
        self.table = None

    def initialize(self) -> None:
        try:
            import lancedb
            import pyarrow as pa
        except ImportError:
            raise ImportError("lancedb and pyarrow are required.")

        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), settings.EMBEDDING_DIM)),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("source", pa.string()),
            pa.field("modality", pa.string()),
        ])

        if self.table_name in self.db.table_names():
            self.table = self.db.open_table(self.table_name)
        else:
            self.table = self.db.create_table(self.table_name, schema=schema)
            
        logger.info(f"LanceDB connected to table '{self.table_name}' ({self.table.count_rows()} records) at {self.db_path}")

    def count(self) -> int:
        if self.table is None:
            self.initialize()
        return self.table.count_rows()

    def add(
        self,
        vectors: np.ndarray,
        contents: List[str],
        modalities: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        if self.table is None:
            self.initialize()

        records = []
        for vec, content, mod, meta in zip(vectors, contents, modalities, metadatas):
            title = meta.get("title") or meta.get("timestamp_formatted") or meta.get("source") or str(content)[:60]
            if meta.get("timestamp_formatted"):
                title = f"{meta.get('source_video', 'Video')} @ {meta.get('timestamp_formatted')}"

            records.append({
                "vector": vec.tolist() if hasattr(vec, "tolist") else list(vec),
                "title": str(title),
                "content": str(content),
                "source": str(meta.get("source", "unknown")),
                "modality": str(mod),
            })

        self.table.add(records)
        logger.info(f"Added {len(records)} records to LanceDB table '{self.table_name}'.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        modality: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if self.table is None:
            self.initialize()

        q_list = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        query_builder = self.table.search(q_list)

        if modality:
            query_builder = query_builder.where(f"modality = '{modality.lower()}'")

        raw_results = query_builder.limit(top_k).to_list()

        results = []
        for r in raw_results:
            sim = float(np.dot(query_vector, r["vector"])) if "vector" in r else 0.0
            results.append({
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "source": r.get("source", ""),
                "modality": r.get("modality", "text"),
                "similarity": round(sim, 4),
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results
