"""
Interactive ROI selector.

Opens a window with the first frame of the video and lets the user
draw a rectangle to define the Region of Interest.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def select_roi_from_frame(frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Display the frame in a window and let the user draw a ROI rectangle.

    Returns:
        (x, y, w, h) tuple of the selected ROI, or None if the user
        pressed ESC / closed the window without selecting.
    """
    window_name = "ViTTA - Select ROI (drag rectangle, then press ENTER/SPACE to confirm, ESC to skip)"

    # Scale down for display if the frame is very large
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1600 or h > 900:
        scale = min(1600 / w, 900 / h)
        display_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        display_frame = frame.copy()

    logger.info("Opening ROI selection window. Draw a rectangle and press ENTER to confirm, or ESC to skip.")

    roi = cv2.selectROI(window_name, display_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w_roi, h_roi = roi
    if w_roi == 0 or h_roi == 0:
        logger.info("No ROI selected — using full frame.")
        return None

    # Scale back to original resolution
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w_roi = int(w_roi / scale)
        h_roi = int(h_roi / scale)

    logger.info(f"ROI selected: x={x}, y={y}, w={w_roi}, h={h_roi}")
    return (x, y, w_roi, h_roi)
