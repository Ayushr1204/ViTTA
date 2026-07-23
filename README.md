# ViTTA — Video-based Traffic Trajectory Extraction and Analysis

A computer vision tool for detecting, tracking, and analyzing road users in Indian mixed traffic conditions from pre-recorded mid-block traffic videos.

Built as a capstone project, ViTTA uses **YOLOv8** for vehicle detection and **ByteTrack** for multi-object tracking, producing a structured Excel dataset with per-vehicle trajectories at 1-second intervals.

---

## Features

- **8-class vehicle detection** — Car, Bus, Truck, Auto-rickshaw, Two-wheeler, LCV, Bicycle, Pedestrian (trained on IDD dataset)
- **ByteTrack multi-object tracking** — Kalman-filtered tracking with 3-round association, lost-track recovery, and trajectory interpolation
- **Spatial calibration & Direction** — Pixel-to-metre conversion for real-world speed (km/h) and cardinal direction detection (N/S/E/W)
- **Interactive Web Dashboard** — Live preview, trajectory visualizations, and automatic post-processing analytics (charts & graphs)
- **Comprehensive Data Export** — Wide-format Excel dataset, per-frame CSV, and automatically generated PDF traffic analysis reports
- **Annotated video output** — Bounding boxes, track IDs, class labels, and trajectory trails
- **Frame extraction & preprocessing** — ROI cropping, CLAHE contrast enhancement, denoising, sharpening

---

## Installation

**Prerequisites**: Python 3.10+, [uv](https://docs.astral.sh/uv/) (recommended) or pip

```bash
# Clone the repository
git clone https://github.com/your-username/ViTTA.git
cd ViTTA

# Create virtual environment with Python 3.10
uv venv .venv --python 3.10
# or: python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
uv pip install -e .
# or: pip install -e .
```

---

## Usage

ViTTA provides a unified CLI with two subcommands:

### Full Tracking Pipeline

```bash
python main.py track --video path/to/traffic.mp4 --model IDD_YOLO/best.pt
```

This runs the complete pipeline: YOLO detection → ByteTrack tracking → CSV output → 1-second resampling → Excel export.

**Output files** (in `output/` by default):

| File | Description |
|------|-------------|
| `tracked_output.mp4` | Annotated video with bounding boxes, IDs, and trajectory trails |
| `tracks.csv` | Raw per-frame tracking data with class name, speed, distance |
| `vitta_results.xlsx` | Final dataset — wide-format trajectory (1 row/vehicle), summary, metadata |

**Common options:**

```bash
# With live preview
python main.py track --video video.mp4 --model IDD_YOLO/best.pt --show

# Custom output directory
python main.py track --video video.mp4 --model IDD_YOLO/best.pt -o results

# Lower detection threshold (catches more vehicles)
python main.py track --video video.mp4 --model IDD_YOLO/best.pt --conf 0.15

# Skip video output (faster)
python main.py track --video video.mp4 --model IDD_YOLO/best.pt --no-video
```

### Frame Extraction Only

```bash
python main.py extract --video path/to/traffic.mp4
python main.py extract --video video.mp4 --roi --interval 3
```

### Full CLI Reference

```bash
python main.py --help
python main.py track --help
python main.py extract --help
```

### Web App (Upload Video + Live Preview + CSV Download)

```bash
uvicorn web_app:app --reload
# If port 8000 is busy on your machine:
uvicorn web_app:app --reload --port 8010
```

Web app flow:
- Upload a video from your browser
- Select an optional **Region of Interest (ROI)** and provide an optional **Spatial Calibration** (reference road length in metres)
- Runs full YOLO + ByteTrack on the video with live preview of the processing
- Generates a rich **Analytics Dashboard** upon completion (vehicle composition, speed histograms, traffic density, direction breakdown)
- Provides 1-click downloads for:
  - `.csv` raw tracking data
  - `.mp4` annotated video with trails
  - `.pdf` auto-generated traffic analysis report

---

## Excel Output Format

The `vitta_results.xlsx` workbook contains three sheets:

### Sheet 1 — Trajectory Data (Wide Format)

One row per vehicle, columns expand for each 1-second timestamp:

| Track ID | Class Name | First Seen | Last Seen | t=0s CX | t=0s CY | t=0s Speed | t=0s Dist | t=1s CX | ... |
|----------|-----------|------------|-----------|---------|---------|------------|-----------|---------|-----|
| 1 | Car | 0 | 12 | 450.2 | 320.1 | 0 | 0 | 478.5 | ... |
| 2 | Bus | 3 | 15 | — | — | — | — | — | ... |

### Sheet 2 — Track Summary

| Track ID | Class | Direction | Duration (s) | Total Distance (px) | Avg Speed (px/s) | Max Speed (px/s) | ... (m/s & km/h if calibrated) |
|----------|-------|-----------|-------------|---------------------|------------------|------------------|--------------------------------|
| 1 | Car | Eastbound | 12 | 1543.2 | 128.6 | 195.3 | ... |

### Sheet 3 — Metadata

Video properties, YOLO config, tracker parameters, and processing statistics.

---

## Project Structure

```
ViTTA/
├── main.py                     # Unified CLI entry point
├── track_demo.py               # Standalone tracking script
├── pyproject.toml              # Dependencies and build config
├── IDD_YOLO/
│   ├── best.pt                 # Trained YOLOv8-Large weights
│   └── args.yaml               # Training hyperparameters
└── vitta/
    ├── __init__.py
    ├── class_names.py           # Vehicle class ID ↔ name mapping
    ├── config.py                # Pipeline configuration
    ├── frame_extractor.py       # Video → frames with metadata
    ├── preprocessor.py          # CLAHE, denoise, sharpen pipeline
    ├── roi_selector.py          # Interactive ROI selection
    ├── tracking/
    │   ├── config.py            # ByteTrack hyperparameters
    │   ├── tracker.py           # ByteTracker (3-round association)
    │   ├── tracker_utils.py     # Kalman filter, IoU, Hungarian matching
    │   ├── metrics.py           # Speed and distance computation
    │   ├── resampler.py         # 1-second interval resampling
    │   ├── csv_writer.py        # Per-frame CSV output
    │   └── visualizer.py        # Bounding box and trail rendering
    └── export/
        └── excel_writer.py      # Multi-sheet Excel exporter
```

---

## YOLO Model

The detection model is YOLOv8-Large fine-tuned on the **IDD (Indian Driving Dataset)** detection subset.

| Parameter | Value |
|-----------|-------|
| Architecture | YOLOv8-Large |
| Training epochs | 50 |
| Image size | 720 |
| Batch size | 32 |
| Optimizer | AdamW |
| Classes | Car, Bus, Truck, Auto, 2W, LCV, Bicycle, Pedestrian |

---

## License

This project was developed as a capstone project for academic purposes.
