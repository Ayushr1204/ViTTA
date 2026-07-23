"""
FastAPI web application for ViTTA.

Features:
- Video upload + async processing job
- ROI selection via quadrilateral on first frame
- Live preview of every processed frame
- CSV and annotated video download on completion
"""

from __future__ import annotations

import base64
import csv
import json
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from vitta.tracking.config import TrackerConfig
from vitta.tracking.csv_writer_v2 import AggregatedCSVWriter
from vitta.tracking.tracker import ByteTracker
from vitta.tracking.visualizer import TrackVisualizer

import logging
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
MODEL_DEFAULT = BASE_DIR / "IDD_YOLO" / "best.pt"


@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued | running | done | failed
    message: str = "Waiting to start..."
    progress: float = 0.0
    total_frames: int = 0
    frame_index: int = 0
    processed_frames: int = 0
    active_tracks: int = 0
    unique_tracks: int = 0
    csv_rows: int = 0
    elapsed_sec: float = 0.0
    latest_preview_name: Optional[str] = None
    previews_count: int = 0
    output_dir: Optional[Path] = None
    upload_dir: Optional[Path] = None
    csv_path: Optional[Path] = None
    tracked_video_path: Optional[Path] = None
    pdf_path: Optional[Path] = None
    pixels_per_metre: Optional[float] = None
    error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


app = FastAPI(title="ViTTA Web")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
jobs: Dict[str, JobState] = {}

# Temporary storage for uploaded videos awaiting ROI selection
pending_uploads: Dict[str, Path] = {}


def _parse_int(value: str, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _count_unique_tracks(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    ids = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row.get("track_id")
            if track_id:
                ids.add(track_id)
    return len(ids)


def _build_roi_mask(
    roi_points: List[List[float]],
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """Build a binary mask from a list of polygon vertices (quadrilateral)."""
    if not roi_points or len(roi_points) < 3:
        return None
    pts = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _apply_roi_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply ROI mask — keep only pixels inside the polygon."""
    return cv2.bitwise_and(frame, frame, mask=mask)


def _filter_detections_by_roi(
    detections: List[List[float]],
    mask: np.ndarray,
) -> List[List[float]]:
    """Keep only detections whose center falls inside the ROI mask."""
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        h, w = mask.shape[:2]
        if 0 <= cx < w and 0 <= cy < h and mask[cy, cx] > 0:
            filtered.append(det)
    return filtered


def _run_job(
    job: JobState,
    *,
    video_path: Path,
    model_path: Path,
    interval: int,
    video_fps: Optional[float] = None,
    roi_points: Optional[List[List[float]]] = None,
    pixels_per_metre: Optional[float] = None,
    calib_points: Optional[List[List[float]]] = None,
) -> None:
    import torch
    from ultralytics import YOLO

    # ── Select GPU device (fall back to CPU) ──
    use_half = False
    if torch.cuda.is_available():
        gpu_device = "0"           # ultralytics predict() device string
        torch_device = "cuda:0"    # PyTorch .to() device string
        use_half = True            # FP16 for ~2x speedup on GPU
        logger.info(f"Using device: CUDA ({torch.cuda.get_device_name(0)}) — FP16 enabled")
    else:
        gpu_device = "cpu"
        torch_device = "cpu"
        logger.info("Using device: CPU")

    # ── Frozen best parameters ──
    conf = 0.25
    iou = 0.45
    img_size = 320
    track_high = 0.60
    track_low = 0.10
    track_buffer = 60
    new_track_thresh = 0.70

    t_start = time.perf_counter()
    output_dir = Path(tempfile.mkdtemp(prefix=f"vitta_job_{job.job_id}_"))
    csv_path = output_dir / "tracks.csv"
    tracked_video_path = output_dir / "tracked_output.mp4"
    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    with job.lock:
        job.status = "running"
        job.message = "Loading model..."
        job.output_dir = output_dir
        job.csv_path = csv_path
        job.tracked_video_path = tracked_video_path

    cap = None
    video_writer = None
    try:
        model = YOLO(str(model_path))
        model.to(torch_device)
        # NOTE: Do NOT call model.half() here — the half=True argument
        # in predict() handles FP16 conversion for both model and input.
        # Manually converting weights causes a dtype mismatch error:
        #   "expected mat1 and mat2 to have the same dtype"
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        detected_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = video_fps if video_fps and video_fps > 0 else detected_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        effective_fps = max(fps / max(interval, 1), 1.0)

        # Build ROI mask if user selected one
        roi_mask = _build_roi_mask(roi_points, width, height) if roi_points else None

        tracker_config = TrackerConfig(
            track_high_thresh=track_high,
            track_low_thresh=track_low,
            new_track_thresh=new_track_thresh,
            track_buffer=track_buffer,
            csv_output_path=csv_path,
        )
        tracker = ByteTracker(tracker_config)
        visualizer = TrackVisualizer(tracker_config)

        # Aggregated CSV writer (one row per vehicle)
        agg_writer = AggregatedCSVWriter(
            csv_path,
            fps=fps,
            frame_height=height,
            pixels_per_metre=pixels_per_metre,
            roi_points=roi_points,
            calib_points=calib_points,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(tracked_video_path), fourcc, effective_fps, (width, height))

        frame_idx = 0
        processed_count = 0
        preview_count = 0

        with job.lock:
            job.total_frames = total_frames
            job.message = "Processing video..."

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % interval != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps

            # Apply ROI mask for detection (feed masked frame to YOLO)
            detection_frame = frame
            if roi_mask is not None:
                detection_frame = _apply_roi_mask(frame.copy(), roi_mask)

            results = model.predict(
                detection_frame, conf=conf, iou=iou, imgsz=img_size,
                device=gpu_device, half=use_half, verbose=False,
            )

            raw_detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    # Vectorised extraction — avoids per-box Python loop
                    xyxy_all = boxes.xyxy.cpu().numpy()
                    conf_all = boxes.conf.cpu().numpy()
                    cls_all = boxes.cls.cpu().numpy().astype(int)
                    for i in range(len(xyxy_all)):
                        raw_detections.append([
                            float(xyxy_all[i, 0]), float(xyxy_all[i, 1]),
                            float(xyxy_all[i, 2]), float(xyxy_all[i, 3]),
                            float(conf_all[i]), int(cls_all[i]),
                        ])

            # Additionally filter detections by ROI center
            if roi_mask is not None:
                raw_detections = _filter_detections_by_roi(raw_detections, roi_mask)

            tracked_objects = tracker.update(raw_detections, processed_count)

            # Draw on the ORIGINAL (unmasked) frame so annotations look natural
            annotated = visualizer.draw(frame.copy(), tracked_objects)

            # Draw ROI polygon outline on annotated frame
            if roi_points and len(roi_points) >= 3:
                roi_pts = np.array(roi_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=2)

            agg_writer.record_frame(processed_count, timestamp, tracked_objects)
            video_writer.write(annotated)

            # Save preview every 5th processed frame (balance I/O vs freshness)
            if processed_count % 5 == 0:
                preview_name = f"preview_{processed_count:06d}.jpg"
                preview_path = previews_dir / preview_name
                # Higher-quality preview for sharper display
                preview_w = 960
                scale = preview_w / annotated.shape[1]
                preview_h = int(annotated.shape[0] * scale)
                small = cv2.resize(annotated, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(preview_path), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
                preview_count += 1
                with job.lock:
                    job.latest_preview_name = preview_name

            processed_count += 1
            frame_idx += 1
            elapsed = time.perf_counter() - t_start
            with job.lock:
                job.frame_index = frame_idx
                job.processed_frames = processed_count
                job.active_tracks = len(tracked_objects)
                job.unique_tracks = tracker.total_tracks_created
                job.csv_rows = agg_writer.unique_tracks
                job.progress = min(frame_idx / total_frames, 1.0) if total_frames > 0 else 0.0
                job.elapsed_sec = elapsed
                job.previews_count = preview_count
                job.latest_preview_name = preview_name
                job.message = "Processing video..."

        # Finalise aggregated CSV (one row per vehicle)
        agg_writer.finalise()

        unique_tracks = agg_writer.rows_written
        csv_rows = agg_writer.rows_written
        elapsed = time.perf_counter() - t_start

        # ── Generate PDF report ───────────────────────────────────────
        pdf_path = output_dir / "vitta_report.pdf"
        try:
            from vitta.export.pdf_report import generate_report
            pdf_csv_data = []
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pdf_csv_data.append(row)
            generate_report(
                csv_data=pdf_csv_data,
                output_path=pdf_path,
                job_stats={
                    "elapsed_sec": elapsed,
                    "processed_frames": processed_count,
                    "csv_rows": csv_rows,
                    "pixels_per_metre": pixels_per_metre,
                },
            )
        except Exception as pdf_exc:
            logger.warning(f"PDF report generation failed: {pdf_exc}", exc_info=True)
            pdf_path = None

        with job.lock:
            job.status = "done"
            job.progress = 1.0
            job.unique_tracks = unique_tracks
            job.csv_rows = csv_rows
            job.elapsed_sec = elapsed
            job.pdf_path = pdf_path
            job.message = "Processing complete."
    except Exception as exc:
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
            job.message = "Job failed."
    finally:
        if cap is not None:
            cap.release()
        if video_writer is not None:
            video_writer.release()
        if job.upload_dir is not None:
            shutil.rmtree(job.upload_dir, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    # Starlette's current signature expects request first, then template name.
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.post("/api/upload-for-roi")
async def upload_for_roi(video: UploadFile = File(...)):
    """
    Upload a video and return its first frame as a base64-encoded JPEG.
    The video is saved temporarily for the subsequent /api/jobs call.
    """
    upload_dir = Path(tempfile.mkdtemp(prefix="vitta_upload_"))
    video_path = upload_dir / (video.filename or "upload.mp4")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    # Extract first frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Cannot open video file.")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Cannot read first frame.")

    # Encode first frame as JPEG base64
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    # Store the upload path keyed by a temporary token
    token = uuid.uuid4().hex[:16]
    pending_uploads[token] = video_path

    h, w = frame.shape[:2]
    return JSONResponse({
        "token": token,
        "first_frame": f"data:image/jpeg;base64,{b64}",
        "width": w,
        "height": h,
    })


@app.post("/api/jobs")
async def create_job(
    token: str = Form(...),
    interval: str = Form("1"),
    video_fps: str = Form(""),
    roi_points: str = Form("[]"),
    pixels_per_metre: str = Form(""),
    calib_points: str = Form("[]"),
):
    video_path = pending_uploads.pop(token, None)
    if video_path is None or not video_path.exists():
        raise HTTPException(status_code=400, detail="Upload token expired or invalid. Please re-upload.")

    model = MODEL_DEFAULT
    if not model.exists():
        raise HTTPException(status_code=400, detail=f"Model file not found: {model}")

    # Parse ROI points
    try:
        roi = json.loads(roi_points)
    except (json.JSONDecodeError, TypeError):
        roi = []

    # Parse calibration points
    try:
        calib_pts = json.loads(calib_points)
    except (json.JSONDecodeError, TypeError):
        calib_pts = []

    # Parse calibration factor
    ppm = None
    if pixels_per_metre.strip():
        ppm_val = _parse_float(pixels_per_metre, 0.0)
        if ppm_val > 0:
            ppm = ppm_val

    # Parse video FPS override
    fps_override = None
    if video_fps.strip():
        fps_val = _parse_float(video_fps, 0.0)
        if fps_val > 0:
            fps_override = fps_val

    upload_dir = video_path.parent

    job_id = uuid.uuid4().hex[:12]
    job = JobState(job_id=job_id, upload_dir=upload_dir, pixels_per_metre=ppm)
    jobs[job_id] = job

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job": job,
            "video_path": video_path,
            "model_path": model,
            "interval": max(1, _parse_int(interval, 1)),
            "video_fps": fps_override,
            "roi_points": roi if roi else None,
            "pixels_per_metre": ppm,
            "calib_points": calib_pts if calib_pts else None,
        },
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    with job.lock:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "progress": job.progress,
            "frame_index": job.frame_index,
            "total_frames": job.total_frames,
            "processed_frames": job.processed_frames,
            "active_tracks": job.active_tracks,
            "unique_tracks": job.unique_tracks,
            "csv_rows": job.csv_rows,
            "elapsed_sec": job.elapsed_sec,
            "previews_count": job.previews_count,
            "pixels_per_metre": job.pixels_per_metre,
            "latest_preview_url": (
                f"/api/jobs/{job_id}/preview/{job.latest_preview_name}"
                if job.latest_preview_name
                else None
            ),
            "csv_download_url": f"/api/jobs/{job_id}/download/csv" if job.status == "done" else None,
            "video_download_url": f"/api/jobs/{job_id}/download/video" if job.status == "done" else None,
            "pdf_download_url": (
                f"/api/jobs/{job_id}/download/pdf"
                if job.status == "done" and job.pdf_path is not None and job.pdf_path.exists()
                else None
            ),
            "error": job.error,
        }


@app.get("/api/jobs/{job_id}/preview/{preview_name}")
def preview_image(job_id: str, preview_name: str):
    job = jobs.get(job_id)
    if job is None or job.output_dir is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = job.output_dir / "previews" / preview_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/download/csv")
def download_csv(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.csv_path is None or not job.csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not available")
    return FileResponse(job.csv_path, media_type="text/csv", filename="tracks.csv")


@app.get("/api/jobs/{job_id}/download/video")
def download_video(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.tracked_video_path is None or not job.tracked_video_path.exists():
        raise HTTPException(status_code=404, detail="Video not available")
    return FileResponse(job.tracked_video_path, media_type="video/mp4", filename="tracked_output.mp4")


@app.get("/api/jobs/{job_id}/download/pdf")
def download_pdf(job_id: str):
    job = jobs.get(job_id)
    if job is None or job.pdf_path is None or not job.pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not available")
    return FileResponse(job.pdf_path, media_type="application/pdf", filename="vitta_report.pdf")


@app.get("/api/jobs/{job_id}/analytics")
def job_analytics(job_id: str):
    """
    Compute analytics from the aggregated CSV (one row per vehicle).
    Returns class distribution, speed histogram, traffic density
    over time, direction breakdown, and summary statistics.
    """
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or job.csv_path is None or not job.csv_path.exists():
        raise HTTPException(status_code=400, detail="Job not finished or CSV unavailable")

    # ── Read aggregated CSV (one row per vehicle) ─────────────────────
    rows = []
    fieldnames = []
    with open(job.csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows.append(row)

    if not rows:
        return JSONResponse({"error": "No data in CSV"})

    from collections import defaultdict
    import math

    # ── Detect calibrated vs pixel units from column names ────────────
    is_calibrated = any("m_per_s" in f for f in fieldnames)
    spd_prefix = "avg_speed_m_per_s" if is_calibrated else "avg_speed_px_per_s"
    spd_label = "m/s" if is_calibrated else "px/s"

    def _find_col(row_dict, prefix, default=""):
        for k in row_dict:
            if k.startswith(prefix):
                return k
        return default

    def _get_float(row_dict, prefix, default=0.0):
        col = _find_col(row_dict, prefix)
        if not col:
            return default
        try:
            return float(row_dict[col])
        except (TypeError, ValueError):
            return default

    def _find_temporal_x(row_dict, sample_idx):
        """Check if temporal sample *sample_idx* exists (any x column)."""
        prefix = f"t{sample_idx}_x"
        for k in row_dict:
            if k.startswith(prefix):
                val = row_dict[k]
                if val != "" and val is not None:
                    return True
        return False

    def _count_temporal_samples(row_dict):
        n = 0
        while _find_temporal_x(row_dict, n):
            n += 1
        return n

    # ── 1. Class distribution ─────────────────────────────────────────
    class_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        class_counts[row.get("class_name", "Unknown")] += 1
    class_labels = sorted(class_counts.keys())
    class_values = [class_counts[c] for c in class_labels]

    # ── 2. Speed distribution (histogram with 1 m/s bins) ─────────────
    avg_speeds: List[float] = []
    for row in rows:
        s = _get_float(row, spd_prefix)
        if s > 0:
            avg_speeds.append(s)

    # Peak speed = max of avg_speed across all vehicles
    inst_prefix = "inst_speed_m_per_s" if is_calibrated else "inst_speed_px_per_s"
    peak_speed = max(avg_speeds) if avg_speeds else 0.0

    # Histogram uses per-vehicle avg speeds (one data point per vehicle)
    hist_speeds = list(avg_speeds)

    # Fixed 1 m/s bin size — covers from 0 to ceil(peak_speed)
    hist_max = math.ceil(peak_speed) if peak_speed > 0 else 1
    bin_width = 1.0  # 1 m/s bins
    num_bins = max(1, int(hist_max / bin_width))
    speed_bins_f = [round(i * bin_width, 1) for i in range(num_bins + 1)]

    speed_hist = [0] * (len(speed_bins_f) - 1)
    for s in hist_speeds:
        bin_idx = int(s / bin_width)
        if bin_idx >= len(speed_hist):
            bin_idx = len(speed_hist) - 1
        if bin_idx >= 0:
            speed_hist[bin_idx] += 1
    speed_bin_labels = [f"{speed_bins_f[i]}-{speed_bins_f[i+1]}" for i in range(len(speed_bins_f) - 1)]

    # ── 3. Traffic density over time ──────────────────────────────────
    vehicle_intervals: List[Tuple[float, float]] = []
    for row in rows:
        try:
            t_start = float(row.get("timestamp_first_seen", 0))
        except (TypeError, ValueError):
            t_start = 0.0
        num_samples = _count_temporal_samples(row)
        t_end = t_start + max(0, num_samples - 1) * 0.5
        vehicle_intervals.append((t_start, t_end))

    density_labels: List[int] = []
    density_values: List[int] = []
    if vehicle_intervals:
        global_start = int(min(t[0] for t in vehicle_intervals))
        global_end = int(max(t[1] for t in vehicle_intervals)) + 1
        density_labels = list(range(global_start, global_end + 1))
        for t in density_labels:
            count = sum(1 for (ts, te) in vehicle_intervals if ts <= t + 1 and te >= t)
            density_values.append(count)

    # ── 4. Direction breakdown ────────────────────────────────────────
    direction_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        d = row.get("direction", "Unknown")
        direction_counts[d] += 1
    dir_labels = sorted(direction_counts.keys())
    dir_values = [direction_counts[d] for d in dir_labels]

    # ── 5. Behaviour distribution ─────────────────────────────────────
    behaviour_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        b = row.get("behavior_class", "Unknown")
        behaviour_counts[b] += 1
    beh_labels = sorted(behaviour_counts.keys())
    beh_values = [behaviour_counts[b] for b in beh_labels]

    # ── 6. Congestion Index over time ─────────────────────────────────
    # Congestion = (active vehicles / capacity_estimate) weighted by
    # inverse of average speed.  Simplified: density_count * (1 / avg_inst_speed)
    # Normalised to 0-100 scale.
    congestion_labels: List[int] = []
    congestion_values: List[float] = []
    if density_labels and density_values:
        max_density = max(density_values) if density_values else 1
        for idx, t in enumerate(density_labels):
            d = density_values[idx] if idx < len(density_values) else 0
            # Congestion is proportional to density
            raw = (d / max(max_density, 1)) * 100
            congestion_values.append(round(raw, 1))
        congestion_labels = density_labels

    # ── 7. Acceleration per class ──────────────────────────────────────
    accel_prefix_lin = "accel_linear_m_per_s2" if is_calibrated else "accel_linear_px_per_s2"
    accel_prefix_lat = "accel_lateral_m_per_s2" if is_calibrated else "accel_lateral_px_per_s2"
    class_lin_accels: Dict[str, List[float]] = defaultdict(list)
    class_lat_accels: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        cls = row.get("class_name", "Unknown")
        i = 0
        while _find_temporal_x(row, i):
            la_lin = _get_float(row, f"t{i}_{accel_prefix_lin}")
            la_lat = _get_float(row, f"t{i}_{accel_prefix_lat}")
            if la_lin != 0:
                class_lin_accels[cls].append(max(-50, min(50, la_lin)))
            if la_lat != 0:
                class_lat_accels[cls].append(max(-50, min(50, la_lat)))
            i += 1

    # Build per-class avg |acceleration| stats for grouped bar chart
    all_accel_classes = sorted(set(list(class_lin_accels.keys()) + list(class_lat_accels.keys())))
    accel_by_class = {}
    for cls in all_accel_classes:
        lin_vals = class_lin_accels.get(cls, [])
        lat_vals = class_lat_accels.get(cls, [])
        accel_by_class[cls] = {
            "avg_linear": round(sum(abs(v) for v in lin_vals) / len(lin_vals), 3) if lin_vals else 0,
            "max_linear": round(max(abs(v) for v in lin_vals), 3) if lin_vals else 0,
            "avg_lateral": round(sum(abs(v) for v in lat_vals) / len(lat_vals), 3) if lat_vals else 0,
            "max_lateral": round(max(abs(v) for v in lat_vals), 3) if lat_vals else 0,
        }

    # ── 8. Speed vs Headway scatter ───────────────────────────────────
    scatter_speeds: List[float] = []
    scatter_headways: List[float] = []
    scatter_classes: List[str] = []
    hw_prefix = "dist_headway_m" if is_calibrated else "dist_headway_px"
    for row in rows:
        vehicle_class = row.get("class_name", "Unknown")
        i = 0
        while _find_temporal_x(row, i):
            s = _get_float(row, f"t{i}_{inst_prefix}")
            h = _get_float(row, f"t{i}_time_headway_s")
            if s > 0 and h > 0 and h < 60:
                scatter_speeds.append(round(s, 2))
                scatter_headways.append(round(h, 2))
                scatter_classes.append(vehicle_class)
            i += 1

    # ── 9. Class-wise speed comparison ────────────────────────────────
    class_speeds: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        cls = row.get("class_name", "Unknown")
        s = _get_float(row, spd_prefix)
        if s > 0:
            class_speeds[cls].append(s)
    # Build per-class stats for bar chart
    class_speed_data = {}
    for cls in sorted(class_speeds.keys()):
        vals = class_speeds[cls]
        if vals:
            class_speed_data[cls] = {
                "avg": round(sum(vals) / len(vals), 2),
                "min": round(min(vals), 2),
                "max": round(max(vals), 2),
                "count": len(vals),
            }

    # ── 10. Summary stats ─────────────────────────────────────────────
    total_vehicles = len(rows)
    avg_speed = round(sum(avg_speeds) / len(avg_speeds), 1) if avg_speeds else 0

    durations = []
    for (ts, te) in vehicle_intervals:
        dur = te - ts
        if dur > 0:
            durations.append(dur)
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0

    total_time = (
        max(t[1] for t in vehicle_intervals) - min(t[0] for t in vehicle_intervals)
    ) if vehicle_intervals else 1
    vehicles_per_min = round(total_vehicles / (total_time / 60), 1) if total_time > 0 else 0

    peak_congestion = round(max(congestion_values), 1) if congestion_values else 0
    avg_congestion = round(sum(congestion_values) / len(congestion_values), 1) if congestion_values else 0

    # Average time headway across all samples
    all_time_headways: List[float] = []
    for row in rows:
        i = 0
        while _find_temporal_x(row, i):
            th = _get_float(row, f"t{i}_time_headway_s")
            if 0 < th < 120:
                all_time_headways.append(th)
            i += 1
    avg_headway = round(sum(all_time_headways) / len(all_time_headways), 1) if all_time_headways else 0

    return JSONResponse({
        "class_distribution": {"labels": class_labels, "values": class_values},
        "speed_histogram": {"labels": speed_bin_labels, "values": speed_hist},
        "traffic_density": {"labels": density_labels, "values": density_values},
        "direction_breakdown": {"labels": dir_labels, "values": dir_values},
        "behaviour_distribution": {"labels": beh_labels, "values": beh_values},
        "congestion_index": {"labels": congestion_labels, "values": congestion_values},
        "accel_by_class": accel_by_class,
        "speed_vs_headway": {
            "speeds": scatter_speeds,
            "headways": scatter_headways,
            "classes": scatter_classes,
        },
        "class_speed_comparison": class_speed_data,
        "summary": {
            "total_vehicles": total_vehicles,
            "avg_speed": avg_speed,
            "peak_speed": round(peak_speed, 1),
            "avg_duration_sec": avg_duration,
            "vehicles_per_min": vehicles_per_min,
            "speed_unit": spd_label,
            "peak_congestion": peak_congestion,
            "avg_congestion": avg_congestion,
            "avg_headway_sec": avg_headway,
        },
    })

