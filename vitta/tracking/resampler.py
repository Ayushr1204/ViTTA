"""
1-second interval resampler for track data.

Takes per-frame tracking records and produces one record per track
per second, with interpolated positions and computed speed / distance.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from vitta.class_names import class_name
from vitta.tracking.metrics import (
    compute_instantaneous_speed,
    euclidean_distance,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FrameRecord:
    """One observation of a track at a specific frame."""
    frame_id: int
    timestamp: float        # seconds from video start
    track_id: int
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    centroid: Tuple[float, float]              # cx, cy


@dataclass
class ResampledRecord:
    """One row in the 1-second-interval output dataset."""
    track_id: int
    class_id: int
    class_name: str
    timestamp_sec: float            # rounded to nearest integer second
    cx_px: float
    cy_px: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    speed_px_per_sec: float
    cumulative_distance_px: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "timestamp_sec": round(self.timestamp_sec, 2),
            "cx_px": round(self.cx_px, 2),
            "cy_px": round(self.cy_px, 2),
            "bbox_x1": round(self.bbox_x1, 2),
            "bbox_y1": round(self.bbox_y1, 2),
            "bbox_x2": round(self.bbox_x2, 2),
            "bbox_y2": round(self.bbox_y2, 2),
            "speed_px_per_sec": round(self.speed_px_per_sec, 2),
            "cumulative_distance_px": round(self.cumulative_distance_px, 2),
            "confidence": round(self.confidence, 4),
        }


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b at parameter t ∈ [0, 1]."""
    return a + (b - a) * t


def _interpolate_record_at(
    before: FrameRecord,
    after: FrameRecord,
    target_time: float,
) -> Tuple[Tuple[float, float], Tuple[float, float, float, float], float]:
    """
    Linearly interpolate centroid, bbox, and confidence at *target_time*
    between two frame records.

    Returns (centroid, bbox, confidence).
    """
    if after.timestamp == before.timestamp:
        t = 0.0
    else:
        t = (target_time - before.timestamp) / (after.timestamp - before.timestamp)
    t = max(0.0, min(1.0, t))

    cx = _lerp(before.centroid[0], after.centroid[0], t)
    cy = _lerp(before.centroid[1], after.centroid[1], t)

    bx1 = _lerp(before.bbox[0], after.bbox[0], t)
    by1 = _lerp(before.bbox[1], after.bbox[1], t)
    bx2 = _lerp(before.bbox[2], after.bbox[2], t)
    by2 = _lerp(before.bbox[3], after.bbox[3], t)

    conf = _lerp(before.confidence, after.confidence, t)

    return (cx, cy), (bx1, by1, bx2, by2), conf


# ═══════════════════════════════════════════════════════════════════════
# Main resampler
# ═══════════════════════════════════════════════════════════════════════

class TrackResampler:
    """
    Resample per-frame track data to 1-second intervals.

    Workflow::

        resampler = TrackResampler()

        # Feed frame records as they are produced (or after loading CSV)
        resampler.add_record(record)

        # After all frames are processed:
        results = resampler.resample()
    """

    def __init__(self, interval_sec: float = 1.0):
        self.interval_sec = interval_sec
        # track_id → list of FrameRecord, sorted by timestamp
        self._tracks: Dict[int, List[FrameRecord]] = {}

    def add_record(self, record: FrameRecord) -> None:
        """Append a per-frame record to the internal buffer."""
        tid = record.track_id
        if tid not in self._tracks:
            self._tracks[tid] = []
        self._tracks[tid].append(record)

    def add_from_tracked_objects(
        self,
        frame_id: int,
        timestamp: float,
        tracked_objects: list,
    ) -> None:
        """
        Convenience method to feed TrackedObject instances directly.

        Args:
            frame_id:  Current frame index.
            timestamp: Current timestamp in seconds.
            tracked_objects: List of TrackedObject from ByteTracker.
        """
        for obj in tracked_objects:
            rec = FrameRecord(
                frame_id=frame_id,
                timestamp=timestamp,
                track_id=obj.track_id,
                class_id=obj.class_id,
                confidence=obj.confidence,
                bbox=(
                    float(obj.bbox[0]),
                    float(obj.bbox[1]),
                    float(obj.bbox[2]),
                    float(obj.bbox[3]),
                ),
                centroid=(float(obj.centroid[0]), float(obj.centroid[1])),
            )
            self.add_record(rec)

    def resample(self) -> List[ResampledRecord]:
        """
        Produce the 1-second-interval dataset for all tracks.

        For each track, we identify every integer-second timestamp that
        falls within [first_seen, last_seen] and interpolate position,
        bbox, and confidence at that instant.

        Returns:
            Sorted list of ResampledRecord (by track_id, then timestamp).
        """
        all_records: List[ResampledRecord] = []

        for track_id in sorted(self._tracks.keys()):
            frames = self._tracks[track_id]
            if not frames:
                continue

            # Sort by timestamp
            frames.sort(key=lambda r: r.timestamp)
            class_id = frames[0].class_id

            t_start = frames[0].timestamp
            t_end = frames[-1].timestamp

            # Integer-second sample points within the track's lifespan
            first_sec = math.ceil(t_start)
            last_sec = math.floor(t_end)
            sample_times = list(range(first_sec, last_sec + 1))

            if not sample_times:
                # Track too short to span a full second — emit one record
                # at the nearest integer second that falls within [t_start, t_end],
                # or the rounded start time if none does.
                nearest_sec = round(t_start)
                sample_times = [nearest_sec]

            prev_centroid: Optional[Tuple[float, float]] = None
            cumulative_dist = 0.0

            for target_sec in sample_times:
                target_time = float(target_sec)

                # Find the bracketing frame records
                centroid, bbox, conf = self._interpolate_at(frames, target_time)

                # Speed
                speed = 0.0
                if prev_centroid is not None:
                    speed = compute_instantaneous_speed(
                        prev_centroid, centroid, self.interval_sec
                    )
                    cumulative_dist += euclidean_distance(prev_centroid, centroid)

                rec = ResampledRecord(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name(class_id),
                    timestamp_sec=target_time,
                    cx_px=centroid[0],
                    cy_px=centroid[1],
                    bbox_x1=bbox[0],
                    bbox_y1=bbox[1],
                    bbox_x2=bbox[2],
                    bbox_y2=bbox[3],
                    speed_px_per_sec=speed,
                    cumulative_distance_px=cumulative_dist,
                    confidence=conf,
                )
                all_records.append(rec)
                prev_centroid = centroid

        logger.info(
            f"Resampled {len(self._tracks)} tracks → "
            f"{len(all_records)} records at {self.interval_sec}s intervals "
            f"(tracks with data: {sum(1 for t in self._tracks.values() if t)})"
        )
        return all_records

    # ── Private ───────────────────────────────────────────────────────

    @staticmethod
    def _interpolate_at(
        frames: List[FrameRecord],
        target_time: float,
    ) -> Tuple[Tuple[float, float], Tuple[float, float, float, float], float]:
        """
        Interpolate centroid/bbox/conf at *target_time* within sorted frames.
        """
        # Exact match
        for f in frames:
            if abs(f.timestamp - target_time) < 1e-6:
                return f.centroid, f.bbox, f.confidence

        # Before first frame — clamp
        if target_time <= frames[0].timestamp:
            f = frames[0]
            return f.centroid, f.bbox, f.confidence

        # After last frame — clamp
        if target_time >= frames[-1].timestamp:
            f = frames[-1]
            return f.centroid, f.bbox, f.confidence

        # Binary-search for the bracketing pair
        lo, hi = 0, len(frames) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if frames[mid].timestamp <= target_time:
                lo = mid
            else:
                hi = mid

        return _interpolate_record_at(frames[lo], frames[hi], target_time)
