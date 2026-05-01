"""
Track visualisation utilities.

Draws bounding boxes, track IDs, confidence scores, and trajectory
trails on video frames.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vitta.class_names import class_name as _get_class_name
from vitta.tracking.config import TrackerConfig
from vitta.tracking.tracker_utils import TrackedObject

logger = logging.getLogger(__name__)


# ── Colour palette ────────────────────────────────────────────────────
# Uses the golden-ratio trick for perceptually distinct colours.

def _id_to_colour(track_id: int) -> Tuple[int, int, int]:
    """
    Map a track ID to a deterministic, visually distinct BGR colour.

    Uses the golden-ratio hue distribution so neighbouring IDs don't
    get similar colours.
    """
    golden_ratio = 0.618033988749895
    hue = ((track_id * golden_ratio) % 1.0) * 180  # OpenCV hue is [0, 180]
    hsv = np.array([[[int(hue), 220, 230]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr[0, 0])


class TrackVisualizer:
    """
    Renders tracked objects onto video frames.

    Usage::

        viz = TrackVisualizer(config)
        annotated = viz.draw(frame, tracked_objects)
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()

    def draw(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        draw_boxes: bool = True,
        draw_labels: bool = True,
        draw_trails: bool = True,
    ) -> np.ndarray:
        """
        Draw all tracking annotations on a frame.

        Args:
            frame:           BGR image (will be modified in-place).
            tracked_objects: List of TrackedObject from ByteTracker.update().
            draw_boxes:      Draw bounding boxes.
            draw_labels:     Draw track ID + class + confidence labels.
            draw_trails:     Draw trajectory trails.

        Returns:
            Annotated frame (same reference as input).
        """
        for obj in tracked_objects:
            colour = _id_to_colour(obj.track_id)

            if draw_boxes:
                self._draw_bbox(frame, obj, colour)

            if draw_labels:
                self._draw_label(frame, obj, colour)

            if draw_trails:
                self._draw_trail(frame, obj, colour)

        return frame

    # ── Private drawing helpers ───────────────────────────────────────

    def _draw_bbox(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw the bounding box rectangle."""
        x1, y1, x2, y2 = obj.bbox.astype(int)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            colour,
            thickness=self.config.line_thickness,
        )

    def _draw_label(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw the track ID, class name, and confidence label."""
        x1, y1 = obj.bbox[:2].astype(int)

        class_name = _get_class_name(obj.class_id)
        label = f"ID:{obj.track_id} {class_name} {obj.confidence:.2f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.config.font_scale
        thickness = max(1, self.config.line_thickness - 1)

        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

        # Background rectangle for readability
        cv2.rectangle(
            frame,
            (x1, y1 - th - baseline - 4),
            (x1 + tw + 4, y1),
            colour,
            cv2.FILLED,
        )

        # Text colour: white or black depending on background brightness
        brightness = 0.299 * colour[2] + 0.587 * colour[1] + 0.114 * colour[0]
        text_colour = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            font,
            scale,
            text_colour,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_trail(
        self,
        frame: np.ndarray,
        obj: TrackedObject,
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw trajectory trail with gradient opacity."""
        trail = obj.trajectory_history
        trail_len = self.config.trajectory_trail_length
        points = trail[-trail_len:] if len(trail) > trail_len else trail

        if len(points) < 2:
            return

        # Draw polyline segments with fading opacity
        n = len(points)
        for i in range(1, n):
            # Alpha increases linearly from 0.3 to 1.0
            alpha = 0.3 + 0.7 * (i / n)
            seg_colour = tuple(int(c * alpha) for c in colour)
            thickness = max(1, int(self.config.line_thickness * alpha))

            pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
            pt2 = (int(points[i][0]), int(points[i][1]))

            cv2.line(frame, pt1, pt2, seg_colour, thickness, cv2.LINE_AA)

        # Draw a small circle at the current position
        cx, cy = int(obj.centroid[0]), int(obj.centroid[1])
        cv2.circle(frame, (cx, cy), 3, colour, cv2.FILLED, cv2.LINE_AA)
