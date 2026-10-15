import logging
from pathlib import Path
from typing import List
from PIL import Image
from .base import BaseProcessor, ProcessedChunk

logger = logging.getLogger(__name__)

# Global EasyOCR reader (lazy-loaded on first image)
_OCR_READER = None

def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            logger.info(f"Initializing EasyOCR reader (gpu={use_gpu})...")
            _OCR_READER = easyocr.Reader(['en'], gpu=use_gpu)
        except Exception as e:
            logger.warning(f"Could not initialize EasyOCR: {e}")
            _OCR_READER = False
    return _OCR_READER if _OCR_READER is not False else None

class ImageProcessor(BaseProcessor):
    """
    Multimodal Image Processor with OCR:
    - Validates image dimensions, formats, and integrity with Pillow.
    - Runs local EasyOCR to detect and extract text from flowcharts, diagrams, and screenshots.
    - Yields an image chunk (for visual vision embeddings) AND an OCR text chunk (for semantic diagram Q&A).
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

        # Extract text via EasyOCR
        ocr_text = ""
        try:
            reader = get_ocr_reader()
            if reader:
                extracted_words = reader.readtext(str(file_path), detail=0)
                if extracted_words:
                    ocr_text = " ".join(extracted_words).strip()
                    logger.info(f"EasyOCR extracted {len(extracted_words)} tokens from {file_path.name}")
        except Exception as ex:
            logger.warning(f"OCR failed for {file_path.name}: {ex}")

        chunks: List[ProcessedChunk] = []

        # 1. Main Image visual chunk
        chunks.append(
            ProcessedChunk(
                content=str(file_path),
                modality="image",
                metadata={
                    "source": str(file_path.name),
                    "file_path": str(file_path),
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "modality": "image",
                    "title": f"Image: {file_path.name}",
                    "ocr_summary": ocr_text[:120] if ocr_text else ""
                }
            )
        )

        # 2. If diagram/flowchart text exists, index an OCR text chunk linked to the image
        if ocr_text and len(ocr_text) > 3:
            structured_content = f"Visual Diagram/Flowchart Content from '{file_path.name}':\n{ocr_text}"
            chunks.append(
                ProcessedChunk(
                    content=structured_content,
                    modality="text",
                    metadata={
                        "source": str(file_path.name),
                        "file_path": str(file_path),
                        "modality": "text",
                        "title": f"Flowchart/Diagram Text from {file_path.name}",
                        "is_ocr": True
                    }
                )
            )

        return chunks
