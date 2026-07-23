"""
Tracker utilities: data structures, Kalman filter, IoU, Hungarian
matching, bbox smoothing, and short-frame interpolation.

All pure functions and self-contained classes — no dependency on the
tracker itself, so they can be unit-tested in isolation.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """
    A single detection from YOLO (or any detector).

    Attributes:
        bbox:       [x1, y1, x2, y2] in pixel coordinates.
        confidence: Detection confidence score ∈ [0, 1].
        class_id:   Integer class label.
    """
    bbox: np.ndarray          # shape (4,)
    confidence: float
    class_id: int

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=np.float64)


class TrackState(enum.IntEnum):
    """Lifecycle states for a tracked object."""
    TENTATIVE = 0   # Just created, not yet confirmed
    CONFIRMED = 1   # Actively tracked with enough evidence
    LOST      = 2   # Temporarily missing (kept alive in buffer)
    DELETED   = 3   # Permanently removed


@dataclass
class TrackedObject:
    """
    Full state of a single tracked vehicle across frames.

    This is the object returned by ByteTracker.update() and consumed
    by the visualiser and CSV writer.
    """
    track_id: int
    class_id: int
    confidence: float
    bbox: np.ndarray                                    # [x1, y1, x2, y2]
    centroid: Tuple[float, float]                       # (cx, cy)
    state: TrackState = TrackState.TENTATIVE
    trajectory_history: List[Tuple[float, float]] = field(default_factory=list)
    age: int = 0                # total frames since track was created
    hits: int = 0               # total frames where a detection was matched
    time_since_update: int = 0  # consecutive frames without a match

    def to_dict(self) -> dict:
        """Serialise to the required output format."""
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
            "centroid": [round(float(self.centroid[0]), 2),
                         round(float(self.centroid[1]), 2)],
        }


# ═══════════════════════════════════════════════════════════════════════
# Kalman filter for bounding-box state estimation
# ═══════════════════════════════════════════════════════════════════════

class KalmanBoxTracker:
    """
    Kalman filter wrapper that tracks a single bounding box.

    State vector (8-dimensional):
        [cx, cy, w, h, ẋ, ẏ, ẇ, ḣ]
        where (cx, cy) is the box centre, (w, h) is width/height,
        and the dot-terms are their respective velocities.

    Measurement vector (4-dimensional):
        [cx, cy, w, h]
    """

    _id_counter: int = 0   # class-level unique-ID generator

    def __init__(self, bbox: np.ndarray):
        """
        Initialise a new Kalman tracker from a detection bbox.

        Args:
            bbox: [x1, y1, x2, y2]
        """
        KalmanBoxTracker._id_counter += 1
        self.id: int = KalmanBoxTracker._id_counter

        self.kf = cv2.KalmanFilter(8, 4)  # 8 state dims, 4 measurement dims

        # ── Transition matrix (constant-velocity model) ──────────────
        # x_{t+1} = F · x_t
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i + 4] = 1.0  # position += velocity

        # ── Measurement matrix ───────────────────────────────────────
        # z = H · x   →   we observe [cx, cy, w, h]
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0

        # ── Noise covariances (hand-tuned for traffic bbox tracking) ─
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        # Velocity components are noisier
        self.kf.processNoiseCov[4, 4] = 5e-2
        self.kf.processNoiseCov[5, 5] = 5e-2
        self.kf.processNoiseCov[6, 6] = 1e-2
        self.kf.processNoiseCov[7, 7] = 1e-2

        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        # High initial uncertainty on velocity
        self.kf.errorCovPost[4, 4] = 10.0
        self.kf.errorCovPost[5, 5] = 10.0
        self.kf.errorCovPost[6, 6] = 10.0
        self.kf.errorCovPost[7, 7] = 10.0

        # ── Initialise state from the first detection ────────────────
        cx, cy, w, h = self._bbox_to_cxcywh(bbox)
        self.kf.statePost = np.array(
            [cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32
        ).reshape(8, 1)

        self.hit_streak: int = 0          # consecutive frames with a match
        self.time_since_update: int = 0   # consecutive frames without match
        self.age: int = 0                 # total frames since creation
        self.hits: int = 1                # total matched frames

    # ── Conversions ───────────────────────────────────────────────────

    @staticmethod
    def _bbox_to_cxcywh(bbox: np.ndarray) -> Tuple[float, float, float, float]:
        """Convert [x1, y1, x2, y2] → (cx, cy, w, h)."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return cx, cy, w, h

    @staticmethod
    def _cxcywh_to_bbox(cx: float, cy: float, w: float, h: float) -> np.ndarray:
        """Convert (cx, cy, w, h) → [x1, y1, x2, y2]."""
        w = max(w, 1.0)
        h = max(h, 1.0)
        return np.array([
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ], dtype=np.float64)

    # ── Public API ────────────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """
        Advance the Kalman state by one time-step.

        Returns:
            Predicted bbox as [x1, y1, x2, y2].
        """
        predicted = self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        cx, cy, w, h = predicted[:4].flatten()
        return self._cxcywh_to_bbox(cx, cy, w, h)

    def update(self, bbox: np.ndarray) -> None:
        """
        Correct the Kalman state with a new matched detection.

        Args:
            bbox: Detected [x1, y1, x2, y2].
        """
        self.time_since_update = 0
        self.hit_streak += 1
        self.hits += 1
        cx, cy, w, h = self._bbox_to_cxcywh(bbox)
        measurement = np.array([cx, cy, w, h], dtype=np.float32).reshape(4, 1)
        self.kf.correct(measurement)

    def get_state(self) -> np.ndarray:
        """
        Return the current estimated bbox as [x1, y1, x2, y2].
        """
        state = self.kf.statePost[:4].flatten()
        cx, cy, w, h = state
        return self._cxcywh_to_bbox(cx, cy, w, h)

    @classmethod
    def reset_id_counter(cls) -> None:
        """Reset the global track-ID counter (useful for tests / reruns)."""
        cls._id_counter = 0


# ═══════════════════════════════════════════════════════════════════════
# IoU computation (vectorised)
# ═══════════════════════════════════════════════════════════════════════

def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """
    Compute the IoU matrix between two sets of bounding boxes.

    Args:
        bboxes_a: (N, 4) array of [x1, y1, x2, y2].
        bboxes_b: (M, 4) array of [x1, y1, x2, y2].

    Returns:
        (N, M) IoU matrix.
    """
    if bboxes_a.size == 0 or bboxes_b.size == 0:
        return np.empty((len(bboxes_a), len(bboxes_b)), dtype=np.float64)

    # Intersection
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0:1].T)  # (N, M)
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1:2].T)
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2:3].T)
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3:4].T)

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    # Union
    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


# ═══════════════════════════════════════════════════════════════════════
# Hungarian (linear) assignment
# ═══════════════════════════════════════════════════════════════════════

def associate_detections_to_tracks(
    iou_matrix: np.ndarray,
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Perform Hungarian matching on an IoU cost matrix.

    Args:
        iou_matrix:    (N, M) matrix of IoU scores.
        iou_threshold: Minimum IoU to accept a match.

    Returns:
        matches:        List of (detection_idx, track_idx) pairs.
        unmatched_dets: Indices of unmatched detections.
        unmatched_trks: Indices of unmatched tracks.
    """
    if iou_matrix.size == 0:
        return (
            [],
            list(range(iou_matrix.shape[0])),
            list(range(iou_matrix.shape[1])),
        )

    # scipy minimises cost; we want to maximise IoU → minimise (1 - IoU)
    cost_matrix = 1.0 - iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int]] = []
    unmatched_dets = set(range(iou_matrix.shape[0]))
    unmatched_trks = set(range(iou_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if iou_matrix[r, c] >= iou_threshold:
            matches.append((r, c))
            unmatched_dets.discard(r)
            unmatched_trks.discard(c)

    return matches, sorted(unmatched_dets), sorted(unmatched_trks)


# ═══════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════

def compute_centroid(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Compute the centroid of a bounding box.

    Args:
        bbox: [x1, y1, x2, y2]

    Returns:
        (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def smooth_bbox(
    old_bbox: np.ndarray,
    new_bbox: np.ndarray,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Exponential moving average for bbox smoothing.

    Args:
        old_bbox: Previous smoothed bbox [x1, y1, x2, y2].
        new_bbox: Current detection bbox.
        alpha:    Weight for the new value (higher = less smoothing).

    Returns:
        Smoothed bbox.
    """
    return alpha * np.asarray(new_bbox) + (1.0 - alpha) * np.asarray(old_bbox)


def interpolate_bboxes(
    bbox_start: np.ndarray,
    bbox_end: np.ndarray,
    num_frames: int,
) -> List[np.ndarray]:
    """
    Linearly interpolate bounding boxes to fill a gap.

    Args:
        bbox_start: Bbox at the start of the gap.
        bbox_end:   Bbox at the end of the gap.
        num_frames: Number of intermediate frames to fill.

    Returns:
        List of interpolated bboxes (one per intermediate frame).
    """
    if num_frames <= 0:
        return []

    start = np.asarray(bbox_start, dtype=np.float64)
    end = np.asarray(bbox_end, dtype=np.float64)
    result = []
    for i in range(1, num_frames + 1):
        t = i / (num_frames + 1)
        result.append(start + t * (end - start))
    return result
