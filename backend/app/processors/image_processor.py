import logging
from pathlib import Path
from typing import List
from PIL import Image
from .base import BaseProcessor, ProcessedChunk

logger = logging.getLogger(__name__)

class ImageProcessor(BaseProcessor):
    """
    Multimodal Image Processor:
    Validates images (.png, .jpg, .jpeg, .webp), verifies integrity with Pillow,
    and captures image dimensions and metadata for vectorization.
    """

    @property
    def supported_modality(self) -> str:
        return "image"

    def process(self, file_path: Path, **kwargs) -> List[ProcessedChunk]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        try:
            with Image.open(file_path) as img:
                width, height = img.size
                img_format = img.format or file_path.suffix.lstrip(".").upper()
        except Exception as e:
            logger.error(f"Failed to open image {file_path}: {e}")
            raise ValueError(f"Corrupt or unsupported image file: {file_path.name}")

        return [
            ProcessedChunk(
                content=str(file_path),
                modality="image",
                metadata={
                    "source": str(file_path.name),
                    "file_path": str(file_path),
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "modality": "image"
                }
            )
        ]
