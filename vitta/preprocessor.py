"""
Frame preprocessing module.

Applies a pipeline of image preprocessing steps to improve YOLO detection
performance on Indian traffic footage:
  1. ROI cropping
  2. Resolution normalization
  3. Contrast enhancement (CLAHE)
  4. Denoising (Gaussian blur)
  5. Sharpening (Unsharp mask)
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

from vitta.config import PipelineConfig

logger = logging.getLogger(__name__)


class FramePreprocessor:
    """
    Applies configurable preprocessing steps to raw video frames
    to prepare them for YOLO inference.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # Pre-create the CLAHE object (reusable across frames)
        self._clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=config.clahe_tile_grid_size,
        )

        # Sharpening kernel (Unsharp-mask style)
        self._sharpen_kernel = np.array(
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]],
            dtype=np.float32,
        )

        logger.info(
            f"Preprocessor initialised — "
            f"ROI: {config.roi or 'full-frame'} | "
            f"CLAHE clip={config.clahe_clip_limit} grid={config.clahe_tile_grid_size} | "
            f"denoise={config.apply_denoise} | sharpen={config.apply_sharpening}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run the full preprocessing pipeline on a single BGR frame."""
        out = frame.copy()

        # 1. Crop to ROI
        out = self._apply_roi(out)

        # 2. Resize if a target resolution is set
        out = self._resize(out)

        # 3. Contrast enhancement via CLAHE on the L channel
        out = self._enhance_contrast(out)

        # 4. Denoise
        if self.config.apply_denoise:
            out = self._denoise(out)

        # 5. Sharpen
        if self.config.apply_sharpening:
            out = self._sharpen(out)

        return out

    # ── Private helpers ───────────────────────────────────────────────

    def _apply_roi(self, frame: np.ndarray) -> np.ndarray:
        """Crop the frame to the configured region-of-interest."""
        roi = self.config.roi
        if roi is None:
            return frame

        x, y, w, h = roi
        h_frame, w_frame = frame.shape[:2]

        # Clamp to frame boundaries
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)

        cropped = frame[y : y + h, x : x + w]
        logger.debug(f"ROI crop: ({x},{y},{w},{h}) → {cropped.shape[:2]}")
        return cropped

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to the target resolution if configured."""
        target = self.config.target_resolution
        if target is None:
            return frame

        tw, th = target
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

    def _enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel of LAB colour space.

        This boosts local contrast without over-saturating colours —
        crucial for outdoor traffic footage with shadows and glare.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to the lightness channel only
        l_enhanced = self._clahe.apply(l_channel)

        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        return result

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """Apply mild Gaussian blur to reduce sensor noise."""
        ksize = self.config.denoise_kernel_size
        return cv2.GaussianBlur(frame, ksize, sigmaX=0)

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        """Apply an unsharp-mask style sharpening filter."""
        return cv2.filter2D(frame, ddepth=-1, kernel=self._sharpen_kernel)
