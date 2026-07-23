"""
Configuration constants for the ViTTA pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class PipelineConfig:
    """Central configuration for the video processing pipeline."""

    # ── Video Extraction ──────────────────────────────────────────────
    source_fps: int = 30            # Expected FPS of the input video
    frame_sample_interval: int = 3  # Extract every Nth frame (30fps / 3 = 10fps effective)

    # ── Preprocessing ─────────────────────────────────────────────────
    # Target resolution for extracted frames (width, height).
    # Set to None to keep original resolution.
    target_resolution: Optional[Tuple[int, int]] = None

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Whether to apply Gaussian blur for denoising
    apply_denoise: bool = True
    denoise_kernel_size: Tuple[int, int] = (3, 3)

    # Whether to sharpen frames after contrast enhancement
    apply_sharpening: bool = True

    # ── ROI (Region of Interest) ──────────────────────────────────────
    # ROI defined as (x, y, width, height) in pixels.
    # Set to None to use the full frame.
    roi: Optional[Tuple[int, int, int, int]] = None

    # ── Output ────────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("frames"))
    save_preprocessed_frames: bool = True  # Save intermediate frames for debugging

    # ── YOLO ──────────────────────────────────────────────────────────
    yolo_model_path: Optional[Path] = None  # Path to custom YOLO weights
    yolo_confidence: float = 0.25
    yolo_iou_threshold: float = 0.45

    # ── Tracking ──────────────────────────────────────────────────────
    # Optional TrackerConfig; import lazily to avoid circular deps.
    # Set to None to skip the tracking stage entirely.
    tracker_config: Optional["TrackerConfig"] = None  # type: ignore[name-defined]

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
