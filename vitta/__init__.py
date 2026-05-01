"""
ViTTA - Video-based Traffic Trajectory extraction and Analysis tool
for Indian mixed traffic conditions.
"""

__version__ = "0.1.0"

# Class name mapping (always available)
from vitta.class_names import CLASS_NAMES, class_name

# Tracking sub-package
from vitta.tracking import ByteTracker, TrackerConfig, TrackVisualizer, TrackCSVWriter
from vitta.tracking import TrackResampler, ResampledRecord

# Export sub-package (requires openpyxl)
try:
    from vitta.export import ExcelExporter
except ImportError:
    ExcelExporter = None  # openpyxl not installed
