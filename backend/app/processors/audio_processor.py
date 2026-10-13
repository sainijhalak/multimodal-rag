import logging
from pathlib import Path
from typing import List
import soundfile as sf
from .base import BaseProcessor, ProcessedChunk
from ..config import settings

logger = logging.getLogger(__name__)

class AudioProcessor(BaseProcessor):
    """
    Multimodal Audio Processor:
    Handles audio files (.wav, .mp3, .ogg, .flac).
    Ensures Google EmbeddingGemma-2 audio requirements:
      - 16,000 Hz sample rate
      - Single channel (Mono)
    """

    def __init__(self, target_sample_rate: int = settings.AUDIO_SAMPLE_RATE):
        self.target_sample_rate = target_sample_rate

    @property
    def supported_modality(self) -> str:
        return "audio"

    def process(self, file_path: Path, **kwargs) -> List[ProcessedChunk]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        processed_path = file_path

        try:
            # Inspect audio properties using soundfile
            info = sf.info(str(file_path))
            
            # If not 16kHz or not mono, resample using librosa
            if info.samplerate != self.target_sample_rate or info.channels != 1:
                try:
                    import librosa
                    logger.info(f"Resampling audio {file_path.name} from {info.samplerate}Hz/{info.channels}ch to {self.target_sample_rate}Hz mono")
                    y, sr = librosa.load(str(file_path), sr=self.target_sample_rate, mono=True)
                    resampled_path = file_path.parent / f"{file_path.stem}_16k_mono.wav"
                    sf.write(str(resampled_path), y, self.target_sample_rate)
                    processed_path = resampled_path
                except Exception as ex:
                    logger.warning(f"Could not resample audio: {ex}. Using original file.")
        except Exception as e:
            logger.warning(f"Audio inspection failed: {e}. Passing file as-is.")

        return [
            ProcessedChunk(
                content=str(processed_path),
                modality="audio",
                metadata={
                    "source": str(file_path.name),
                    "file_path": str(processed_path),
                    "sample_rate": self.target_sample_rate,
                    "channels": 1,
                    "modality": "audio"
                }
            )
        ]
