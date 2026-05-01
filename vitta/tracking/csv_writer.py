"""
CSV writer for tracked object data.

Writes one row per tracked object per frame with the schema:
    frame_id, timestamp, track_id, class_id, class_name, confidence,
    x1, y1, x2, y2, cx, cy, speed_px_per_sec, cumulative_distance_px
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vitta.class_names import class_name as get_class_name
from vitta.tracking.tracker_utils import TrackedObject
from vitta.tracking.metrics import compute_instantaneous_speed, euclidean_distance

logger = logging.getLogger(__name__)

# Column header for the output CSV
_CSV_HEADER = [
    "frame_id",
    "timestamp",
    "track_id",
    "class_id",
    "class_name",
    "confidence",
    "x1", "y1", "x2", "y2",
    "cx", "cy",
    "speed_px_per_sec",
    "cumulative_distance_px",
]


class TrackCSVWriter:
    """
    Writes tracked object data to a CSV file.

    Supports context-manager usage::

        with TrackCSVWriter("output/tracks.csv") as writer:
            writer.write_frame(frame_id, timestamp, tracked_objects)

    Or manual open/close::

        writer = TrackCSVWriter("output/tracks.csv")
        writer.write_frame(frame_id, timestamp, tracked_objects)
        writer.close()
    """

    def __init__(self, output_path: str | Path):
        """
        Open the CSV file for writing and emit the header row.

        Args:
            output_path: Destination file path. Parent directories will
                         be created if they don't exist.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = open(self.output_path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_CSV_HEADER)
        self._rows_written: int = 0

        # Per-track state for speed/distance computation
        self._prev_centroid: Dict[int, Tuple[float, float]] = {}
        self._prev_timestamp: Dict[int, float] = {}
        self._cumulative_distance: Dict[int, float] = {}

        logger.info(f"CSV writer opened: {self.output_path}")

    # ── Writing ───────────────────────────────────────────────────────

    def write_frame(
        self,
        frame_id: int,
        timestamp: float,
        tracked_objects: List[TrackedObject],
    ) -> None:
        """
        Write one row per tracked object for a given frame.

        Args:
            frame_id:        Integer frame index.
            timestamp:       Frame timestamp in seconds.
            tracked_objects:  List of TrackedObject from ByteTracker.
        """
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            cx, cy = obj.centroid
            tid = obj.track_id

            # Compute speed and distance
            speed = 0.0
            if tid in self._prev_centroid:
                dt = timestamp - self._prev_timestamp[tid]
                speed = compute_instantaneous_speed(
                    self._prev_centroid[tid], (cx, cy), dt
                )
                self._cumulative_distance[tid] += euclidean_distance(
                    self._prev_centroid[tid], (cx, cy)
                )
            else:
                self._cumulative_distance[tid] = 0.0

            self._prev_centroid[tid] = (cx, cy)
            self._prev_timestamp[tid] = timestamp
            cum_dist = self._cumulative_distance[tid]

            self._writer.writerow([
                frame_id,
                f"{timestamp:.4f}",
                obj.track_id,
                obj.class_id,
                get_class_name(obj.class_id),
                f"{obj.confidence:.4f}",
                f"{x1:.2f}",
                f"{y1:.2f}",
                f"{x2:.2f}",
                f"{y2:.2f}",
                f"{cx:.2f}",
                f"{cy:.2f}",
                f"{speed:.2f}",
                f"{cum_dist:.2f}",
            ])
            self._rows_written += 1

            # Auto-flush periodically for crash safety on long runs
            if self._rows_written % 500 == 0:
                self._file.flush()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def flush(self) -> None:
        """Force-write buffered data to disk."""
        if self._file and not self._file.closed:
            self._file.flush()

    def close(self) -> None:
        """Flush and close the file."""
        if self._file and not self._file.closed:
            self._file.close()
            logger.info(
                f"CSV writer closed: {self.output_path} "
                f"({self._rows_written} rows written)"
            )

    @property
    def rows_written(self) -> int:
        return self._rows_written

    # ── Context manager ───────────────────────────────────────────────

    def __enter__(self) -> "TrackCSVWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
