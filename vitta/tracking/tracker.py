"""
ByteTrack multi-object tracker.

Implements the full ByteTrack algorithm with two-stage association,
Kalman prediction, lost-track recovery, bbox smoothing, trajectory
storage, and short-frame interpolation.

Reference:
    Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
    Every Detection Box", ECCV 2022.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from vitta.tracking.config import TrackerConfig
from vitta.tracking.tracker_utils import (
    Detection,
    KalmanBoxTracker,
    TrackState,
    TrackedObject,
    associate_detections_to_tracks,
    compute_centroid,
    interpolate_bboxes,
    iou_batch,
    smooth_bbox,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Internal track representation
# ═══════════════════════════════════════════════════════════════════════

class _STrack:
    """
    Internal single-track wrapper holding a Kalman tracker, lifecycle
    state, and trajectory history.  Not exposed to callers — the public
    API returns TrackedObject instances.
    """

    def __init__(
        self,
        detection: Detection,
        config: TrackerConfig,
    ):
        self.kalman = KalmanBoxTracker(detection.bbox)
        self.track_id: int = self.kalman.id
        self.class_id: int = detection.class_id
        self.confidence: float = detection.confidence
        self.state: TrackState = TrackState.TENTATIVE

        # Smoothed bbox (initialised to detection bbox)
        self.smooth_bbox: np.ndarray = detection.bbox.copy()

        # Centroid history for trajectory visualisation
        self.trajectory: List[Tuple[float, float]] = [
            compute_centroid(detection.bbox)
        ]

        # Configuration reference
        self._config = config

        # Frame at which the track was last seen (for interpolation)
        self.last_seen_frame: int = -1

        # Store the bbox when we last saw the track (for interpolation)
        self.last_seen_bbox: Optional[np.ndarray] = detection.bbox.copy()

        # Interpolation buffer: frame_id → bbox (filled retroactively)
        self.interpolated_frames: Dict[int, np.ndarray] = {}

    # ── Lifecycle helpers ─────────────────────────────────────────────

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    @property
    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED

    @property
    def time_since_update(self) -> int:
        return self.kalman.time_since_update

    @property
    def hits(self) -> int:
        return self.kalman.hits

    @property
    def age(self) -> int:
        return self.kalman.age

    # ── Kalman wrappers ───────────────────────────────────────────────

    def predict(self) -> np.ndarray:
        """Advance Kalman one step and return predicted bbox."""
        return self.kalman.predict()

    def update(self, detection: Detection, frame_id: int) -> None:
        """
        Match a detection to this track: correct Kalman, smooth bbox,
        append centroid, attempt interpolation if there was a gap.
        """
        # Interpolate the gap if the track was recently lost
        gap = frame_id - self.last_seen_frame if self.last_seen_frame >= 0 else 0
        if (
            gap > 1
            and gap <= self._config.interpolation_max_gap
            and self.last_seen_bbox is not None
        ):
            interp = interpolate_bboxes(self.last_seen_bbox, detection.bbox, gap - 1)
            for i, bbox in enumerate(interp):
                fid = self.last_seen_frame + i + 1
                self.interpolated_frames[fid] = bbox
                cx, cy = compute_centroid(bbox)
                self.trajectory.append((cx, cy))

        # Kalman correction
        self.kalman.update(detection.bbox)

        # Bbox smoothing via EMA
        self.smooth_bbox = smooth_bbox(
            self.smooth_bbox, detection.bbox, self._config.bbox_smoothing_alpha
        )

        # Update metadata
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.last_seen_frame = frame_id
        self.last_seen_bbox = self.smooth_bbox.copy()

        # Append centroid to trajectory
        cx, cy = compute_centroid(self.smooth_bbox)
        self.trajectory.append((cx, cy))

        # Cap trajectory length
        max_len = self._config.trajectory_max_length
        if len(self.trajectory) > max_len:
            self.trajectory = self.trajectory[-max_len:]

        # State transition: tentative → confirmed
        if (
            self.state == TrackState.TENTATIVE
            and self.kalman.hit_streak >= self._config.min_hits
        ):
            self.state = TrackState.CONFIRMED
            logger.debug(f"Track {self.track_id} CONFIRMED after {self.hits} hits")

        # If the track was lost, re-confirm it
        if self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED
            logger.debug(f"Track {self.track_id} RECOVERED")

    def mark_lost(self) -> None:
        """Transition a confirmed track to LOST."""
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def mark_deleted(self) -> None:
        """Permanently remove the track."""
        self.state = TrackState.DELETED

    # ── Export ────────────────────────────────────────────────────────

    def to_tracked_object(self) -> TrackedObject:
        """Export to public TrackedObject dataclass."""
        cx, cy = compute_centroid(self.smooth_bbox)
        return TrackedObject(
            track_id=self.track_id,
            class_id=self.class_id,
            confidence=self.confidence,
            bbox=self.smooth_bbox.copy(),
            centroid=(cx, cy),
            state=self.state,
            trajectory_history=list(self.trajectory),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


# ═══════════════════════════════════════════════════════════════════════
# ByteTracker — main public class
# ═══════════════════════════════════════════════════════════════════════

class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Usage::

        tracker = ByteTracker(TrackerConfig())
        for frame_id, detections in enumerate(detection_source):
            tracked = tracker.update(detections, frame_id)
            for obj in tracked:
                print(obj.to_dict())

    Args:
        config: TrackerConfig with all hyperparameters.
    """

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._tracks: List[_STrack] = []    # active + lost tracks
        self._frame_count: int = 0
        self._total_tracks_created: int = 0

        # Reset the Kalman ID counter so IDs start from 1 each run
        KalmanBoxTracker.reset_id_counter()

        logger.info(
            f"ByteTracker initialised — "
            f"high={self.config.track_high_thresh:.2f} "
            f"low={self.config.track_low_thresh:.2f} "
            f"new={self.config.new_track_thresh:.2f} "
            f"buffer={self.config.track_buffer} "
            f"min_hits={self.config.min_hits}"
        )

    # ── Public API ────────────────────────────────────────────────────

    def update(
        self,
        raw_detections: List[List[float]],
        frame_id: Optional[int] = None,
    ) -> List[TrackedObject]:
        """
        Process one frame of detections and return tracked objects.

        Args:
            raw_detections: List of [x1, y1, x2, y2, confidence, class_id].
                            Can also be a numpy array of shape (N, 6).
            frame_id:       Optional frame index (auto-incremented if None).

        Returns:
            List of TrackedObject for all CONFIRMED tracks in this frame.
        """
        if frame_id is None:
            frame_id = self._frame_count
        self._frame_count = frame_id + 1

        # ── 1. Parse detections ───────────────────────────────────────
        detections = self._parse_detections(raw_detections)

        # ── 2. Split by confidence ────────────────────────────────────
        high_dets: List[Detection] = []
        low_dets: List[Detection] = []

        for det in detections:
            if det.confidence >= self.config.track_high_thresh:
                high_dets.append(det)
            elif det.confidence >= self.config.track_low_thresh:
                low_dets.append(det)
            # Below track_low_thresh → discarded entirely

        # ── 3. Predict all existing tracks ────────────────────────────
        for trk in self._tracks:
            trk.predict()

        # Separate confirmed/tentative from lost tracks
        active_tracks = [t for t in self._tracks if not t.is_deleted]
        confirmed_tracks = [t for t in active_tracks if t.is_confirmed or t.state == TrackState.TENTATIVE]
        lost_tracks = [t for t in active_tracks if t.is_lost]

        # ── 4. First association: high-conf dets ↔ confirmed tracks ──
        matches_1st, unmatched_det_idx, unmatched_trk_idx = self._associate(
            high_dets, confirmed_tracks
        )

        # Apply matches
        for det_idx, trk_idx in matches_1st:
            confirmed_tracks[trk_idx].update(high_dets[det_idx], frame_id)

        # Collect unmatched tracks from first round
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_trk_idx]
        remaining_high_dets = [high_dets[i] for i in unmatched_det_idx]

        # ── 5. Second association: low-conf dets ↔ remaining tracks ──
        # ByteTrack's key innovation: use low-confidence detections to
        # maintain tracks through partial occlusions.
        matches_2nd, unmatched_low_idx, unmatched_trk_idx_2 = self._associate(
            low_dets, remaining_tracks
        )

        for det_idx, trk_idx in matches_2nd:
            remaining_tracks[trk_idx].update(low_dets[det_idx], frame_id)

        # ── 6. Third association: remaining high dets ↔ lost tracks ──
        # Give lost tracks a chance to be recovered by high-confidence
        # detections that didn't match anything active.
        if remaining_high_dets and lost_tracks:
            matches_3rd, unmatched_high_idx_3, _ = self._associate(
                remaining_high_dets, lost_tracks
            )
            for det_idx, trk_idx in matches_3rd:
                lost_tracks[trk_idx].update(remaining_high_dets[det_idx], frame_id)

            # Update remaining_high_dets after lost-track recovery
            remaining_high_dets = [remaining_high_dets[i] for i in unmatched_high_idx_3]

        # ── 7. Handle unmatched tracks → LOST / DELETED ──────────────
        still_unmatched = [remaining_tracks[i] for i in unmatched_trk_idx_2]
        for trk in still_unmatched:
            if trk.time_since_update > self.config.track_buffer:
                trk.mark_deleted()
                logger.debug(f"Track {trk.track_id} DELETED (exceeded buffer)")
            elif trk.is_confirmed:
                trk.mark_lost()

        # Age out lost tracks globally
        for trk in lost_tracks:
            if trk.time_since_update > self.config.track_buffer:
                trk.mark_deleted()

        # ── 8. Spawn new tracks from unmatched high-conf detections ──
        for det in remaining_high_dets:
            if det.confidence >= self.config.new_track_thresh:
                new_track = _STrack(det, self.config)
                new_track.last_seen_frame = frame_id
                self._tracks.append(new_track)
                self._total_tracks_created += 1
                logger.debug(
                    f"New track {new_track.track_id} "
                    f"(class={det.class_id}, conf={det.confidence:.2f})"
                )

        # ── 9. Prune deleted tracks ──────────────────────────────────
        self._tracks = [t for t in self._tracks if not t.is_deleted]

        # ── 10. Build output ─────────────────────────────────────────
        output = [
            trk.to_tracked_object()
            for trk in self._tracks
            if trk.is_confirmed and trk.time_since_update == 0
        ]

        return output

    def get_active_tracks(self) -> List[TrackedObject]:
        """Return all currently confirmed tracks (including those not updated this frame)."""
        return [
            trk.to_tracked_object()
            for trk in self._tracks
            if trk.is_confirmed
        ]

    def get_all_tracks(self) -> List[TrackedObject]:
        """Return all tracks including lost ones (useful for debugging)."""
        return [trk.to_tracked_object() for trk in self._tracks]

    def reset(self) -> None:
        """Clear all tracks and reset counters."""
        self._tracks.clear()
        self._frame_count = 0
        self._total_tracks_created = 0
        KalmanBoxTracker.reset_id_counter()
        logger.info("ByteTracker reset.")

    @property
    def total_tracks_created(self) -> int:
        """Total number of unique tracks created so far."""
        return self._total_tracks_created

    # ── Private helpers ───────────────────────────────────────────────

    def _parse_detections(
        self, raw: List[List[float]]
    ) -> List[Detection]:
        """
        Convert raw detection arrays to Detection objects.

        Args:
            raw: Each element is [x1, y1, x2, y2, confidence, class_id].

        Returns:
            List of Detection instances.
        """
        detections = []
        for item in raw:
            try:
                arr = np.asarray(item, dtype=np.float64)
                if arr.shape[0] < 6:
                    logger.warning(f"Skipping malformed detection (len={arr.shape[0]}): {item}")
                    continue
                det = Detection(
                    bbox=arr[:4],
                    confidence=float(arr[4]),
                    class_id=int(arr[5]),
                )
                detections.append(det)
            except (ValueError, IndexError) as exc:
                logger.warning(f"Skipping invalid detection {item}: {exc}")
        return detections

    def _associate(
        self,
        detections: List[Detection],
        tracks: List[_STrack],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using IoU + Hungarian algorithm.

        Returns:
            (matches, unmatched_det_indices, unmatched_trk_indices)
        """
        if not detections or not tracks:
            return (
                [],
                list(range(len(detections))),
                list(range(len(tracks))),
            )

        # Build bbox arrays
        det_bboxes = np.array([d.bbox for d in detections], dtype=np.float64)
        trk_bboxes = np.array([t.kalman.get_state() for t in tracks], dtype=np.float64)

        # Compute IoU matrix
        iou_matrix = iou_batch(det_bboxes, trk_bboxes)

        # Run Hungarian
        return associate_detections_to_tracks(iou_matrix, 1.0 - self.config.match_thresh)
