"""
Frame extraction module.

Reads a video file and yields every Nth frame along with its metadata
(original frame index, timestamp, etc.).
"""

import cv2
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Generator

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Container for a single extracted frame and its metadata."""
    frame_index: int        # Original 0-based index in the video
    sampled_index: int      # Sequential index among sampled frames (0, 1, 2, …)
    timestamp_sec: float    # Timestamp in seconds (frame_index / fps)
    image: "np.ndarray"     # BGR image array


class FrameExtractor:
    """
    Extracts frames from a video file at a configurable sampling interval.

    For a 30 fps video with sample_interval=3, this yields every 3rd frame
    (frames 0, 3, 6, 9, …), producing an effective 10 fps output.
    """

    def __init__(self, video_path: str | Path, sample_interval: int = 3):
        self.video_path = Path(video_path)
        self.sample_interval = sample_interval

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Open the video and read properties
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")

        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self.total_frames: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width: int = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec: float = self.total_frames / self.fps if self.fps > 0 else 0.0

        self.effective_fps: float = self.fps / self.sample_interval
        self.expected_sampled_frames: int = self.total_frames // self.sample_interval

        logger.info(
            f"Video loaded: {self.video_path.name} | "
            f"{self.width}×{self.height} @ {self.fps:.1f} fps | "
            f"{self.total_frames} frames ({self.duration_sec:.1f}s) | "
            f"Sampling every {self.sample_interval} frames → "
            f"~{self.expected_sampled_frames} output frames @ {self.effective_fps:.1f} fps"
        )

    def extract(self) -> Generator[FrameData, None, None]:
        """
        Yield FrameData objects for every Nth frame in the video.

        Uses CAP_PROP_POS_FRAMES seeking for efficiency on longer videos.
        """
        import numpy as np  # lazy import to keep module-level lightweight

        sampled_idx = 0

        for frame_idx in range(0, self.total_frames, self.sample_interval):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()

            if not ret:
                logger.warning(f"Failed to read frame {frame_idx}, skipping.")
                continue

            timestamp = frame_idx / self.fps if self.fps > 0 else 0.0

            yield FrameData(
                frame_index=frame_idx,
                sampled_index=sampled_idx,
                timestamp_sec=round(timestamp, 4),
                image=frame,
            )
            sampled_idx += 1

        logger.info(f"Extraction complete: {sampled_idx} frames yielded.")

    def get_video_info(self) -> dict:
        """Return a summary dict of video properties."""
        return {
            "file": str(self.video_path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": round(self.duration_sec, 2),
            "sample_interval": self.sample_interval,
            "effective_fps": round(self.effective_fps, 2),
            "expected_sampled_frames": self.expected_sampled_frames,
        }

    def release(self):
        """Release the underlying VideoCapture resource."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.debug("VideoCapture released.")

    def __del__(self):
        self.release()
