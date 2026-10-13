from .base import BaseProcessor, ProcessedChunk
from .text_processor import TextProcessor
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor

PROCESSOR_REGISTRY = {
    ".txt": TextProcessor(),
    ".md": TextProcessor(),
    ".pdf": TextProcessor(), # will extend with PyMuPDF
    ".jpg": ImageProcessor(),
    ".jpeg": ImageProcessor(),
    ".png": ImageProcessor(),
    ".wav": AudioProcessor(),
    ".mp3": AudioProcessor(),
    ".mp4": VideoProcessor(),
    ".mkv": VideoProcessor(),
}

def get_processor_for_file(extension: str) -> BaseProcessor:
    """Returns the processor matching the file extension."""
    processor = PROCESSOR_REGISTRY.get(extension.lower())
    if not processor:
        raise ValueError(f"No processor registered for file extension: {extension}")
    return processor

__all__ = [
    "BaseProcessor",
    "ProcessedChunk",
    "TextProcessor",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "get_processor_for_file",
]
