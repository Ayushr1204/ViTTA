"""
ViTTA Tracking Sub-package.

Provides ByteTrack-based multi-object tracking for vehicle tracking
in dense traffic scenes.
"""

from vitta.tracking.config import TrackerConfig
from vitta.tracking.tracker import ByteTracker
from vitta.tracking.visualizer import TrackVisualizer
from vitta.tracking.csv_writer import TrackCSVWriter
from vitta.tracking.metrics import (
    compute_instantaneous_speed,
    compute_cumulative_distance,
    compute_cumulative_distances,
    euclidean_distance,
)
from vitta.tracking.resampler import TrackResampler, ResampledRecord
from vitta.tracking.tracker_utils import (
    Detection,
    TrackState,
    TrackedObject,
    KalmanBoxTracker,
)

__all__ = [
    "TrackerConfig",
    "ByteTracker",
    "TrackVisualizer",
    "TrackCSVWriter",
    "TrackResampler",
    "ResampledRecord",
    "Detection",
    "TrackState",
    "TrackedObject",
    "KalmanBoxTracker",
    "compute_instantaneous_speed",
    "compute_cumulative_distance",
    "compute_cumulative_distances",
    "euclidean_distance",
]
