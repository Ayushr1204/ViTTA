"""
Tracking configuration.

All ByteTrack hyperparameters are centralised here, with defaults
tuned for dense Indian mixed-traffic scenes.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrackerConfig:
    """
    Configuration for the ByteTrack multi-object tracker.

    Default values are optimised for dense urban traffic with frequent
    occlusions, lane-less driving, and heterogeneous vehicle sizes
    (auto-rickshaws, two-wheelers, buses, etc.).
    """

    # ── ByteTrack association thresholds ──────────────────────────────
    # Detections with confidence >= track_high_thresh enter the FIRST
    # association round (matched against all active tracks).
    track_high_thresh: float = 0.6

    # Detections between track_low_thresh and track_high_thresh enter
    # the SECOND association round — this is ByteTrack's key innovation
    # that recovers partially-occluded objects.
    track_low_thresh: float = 0.1

    # Minimum confidence to spawn a brand-new track.
    new_track_thresh: float = 0.7

    # IoU threshold for the Hungarian matching step.
    # Pairs with IoU below this are rejected even if the solver picks them.
    match_thresh: float = 0.8

    # ── Track lifecycle ───────────────────────────────────────────────
    # How many frames a lost track is kept alive before deletion.
    # At 30 fps, 60 frames ≈ 2 seconds — enough for brief full occlusions.
    track_buffer: int = 60

    # A tentative track must accumulate this many consecutive hits
    # before it is promoted to CONFIRMED.
    min_hits: int = 3

    # ── Kalman / smoothing ────────────────────────────────────────────
    # EMA (exponential moving average) weight for bounding-box smoothing.
    # Higher = more weight on the current detection, lower = smoother.
    bbox_smoothing_alpha: float = 0.7

    # ── Interpolation ─────────────────────────────────────────────────
    # Maximum gap (in frames) that will be filled by linear interpolation
    # when a track is lost and then recovered.
    interpolation_max_gap: int = 10

    # ── Trajectory storage ────────────────────────────────────────────
    # Maximum number of centroid history points to retain per track.
    trajectory_max_length: int = 120

    # ── Visualisation ─────────────────────────────────────────────────
    line_thickness: int = 2
    font_scale: float = 0.6
    trajectory_trail_length: int = 30  # how many past points to draw

    # ── Output ────────────────────────────────────────────────────────
    csv_output_path: Path = field(default_factory=lambda: Path("output/tracks.csv"))

    def __post_init__(self):
        self.csv_output_path = Path(self.csv_output_path)
        self.csv_output_path.parent.mkdir(parents=True, exist_ok=True)
