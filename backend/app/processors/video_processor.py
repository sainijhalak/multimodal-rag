import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image
from .base import BaseProcessor, ProcessedChunk
from ..config import settings

logger = logging.getLogger(__name__)

class VideoProcessor(BaseProcessor):
    """
    Multimodal Video Processor:
    Extracts keyframes at 1 FPS (1 frame per second) from video files (.mp4, .mov, .avi, etc.).
    Indexes each keyframe with its exact temporal timestamp (e.g. 00:05), enabling
    cross-modal text-to-video search and precise timestamp jump-seeking.
    """

    def __init__(self, target_fps: int = settings.VIDEO_FPS, frames_dir: Optional[Path] = None):
        self.target_fps = target_fps
        self.frames_dir = frames_dir or (settings.UPLOADS_DIR / "frames")
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    @property
    def supported_modality(self) -> str:
        return "video"

    def process(self, file_path: Path, **kwargs) -> List[ProcessedChunk]:
        """
        Samples frames at 1 FPS and saves them as preview images with timestamps.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Video file not found: {file_path}")

        try:
            import av
        except ImportError:
            logger.warning("PyAV ('av') not yet imported. Falling back to single-file video descriptor.")
            return [
                ProcessedChunk(
                    content=str(file_path),
                    modality="video",
                    metadata={
                        "source": str(file_path.name),
                        "timestamp_sec": 0.0,
                        "timestamp_formatted": "00:00"
                    }
                )
            ]

        chunks: List[ProcessedChunk] = []
        video_stem = file_path.stem

        try:
            container = av.open(str(file_path))
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"  # Fast seek across keyframes if available

            time_base = float(stream.time_base) if stream.time_base else (1.0 / 30.0)
            avg_rate = float(stream.average_rate) if stream.average_rate else 30.0
            
            # Track sampled seconds to enforce 1 FPS sampling
            last_sampled_sec = -1.0

            for frame in container.decode(stream):
                pts_sec = float(frame.pts * time_base) if frame.pts is not None else float(frame.index / avg_rate)
                
                # Sample at target_fps interval (e.g. 1 frame every 1.0 second)
                if pts_sec - last_sampled_sec >= (1.0 / self.target_fps):
                    last_sampled_sec = pts_sec
                    sec_int = int(pts_sec)
                    mins, secs = divmod(sec_int, 60)
                    timestamp_formatted = f"{mins:02d}:{secs:02d}"

                    frame_filename = f"{video_stem}_t{sec_int:04d}.jpg"
                    frame_path = self.frames_dir / frame_filename

                    # Save keyframe image
                    img = frame.to_image()
                    img.save(frame_path, format="JPEG", quality=85)

                    chunks.append(
                        ProcessedChunk(
                            content=str(frame_path),
                            modality="video",
                            metadata={
                                "source": str(file_path.name),
                                "source_video": str(file_path.name),
                                "frame_path": str(frame_path),
                                "timestamp_sec": round(pts_sec, 2),
                                "timestamp_formatted": timestamp_formatted,
                                "modality": "video"
                            }
                        )
                    )

            container.close()

            # Fallback if no frames decoded
            if not chunks:
                logger.info(f"Container decoding yielded 0 frames with NONKEY; retrying full decode for {file_path.name}")
                container = av.open(str(file_path))
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = "DEFAULT"
                last_sec = -1.0
                for frame in container.decode(stream):
                    pts_sec = float(frame.pts * time_base) if frame.pts is not None else 0.0
                    if pts_sec - last_sec >= 1.0:
                        last_sec = pts_sec
                        sec_int = int(pts_sec)
                        mins, secs = divmod(sec_int, 60)
                        ts_fmt = f"{mins:02d}:{secs:02d}"
                        frame_path = self.frames_dir / f"{video_stem}_t{sec_int:04d}.jpg"
                        img = frame.to_image()
                        img.save(frame_path, format="JPEG", quality=85)
                        chunks.append(
                            ProcessedChunk(
                                content=str(frame_path),
                                modality="video",
                                metadata={
                                    "source": str(file_path.name),
                                    "source_video": str(file_path.name),
                                    "frame_path": str(frame_path),
                                    "timestamp_sec": round(pts_sec, 2),
                                    "timestamp_formatted": ts_fmt,
                                    "modality": "video"
                                }
                            )
                        )
                container.close()

        except Exception as e:
            logger.error(f"Error processing video {file_path}: {e}")
            raise e

        logger.info(f"Extracted {len(chunks)} keyframes at {self.target_fps} FPS from {file_path.name}")
        return chunks
