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
from vitta.tracking.csv_writer import TrackCSVWriter
from vitta.tracking.tracker import ByteTracker
from vitta.tracking.visualizer import TrackVisualizer


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
    roi_points: Optional[List[List[float]]] = None,
) -> None:
    from ultralytics import YOLO

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
    csv_writer = None
    video_writer = None
    try:
        model = YOLO(str(model_path))
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
        csv_writer = TrackCSVWriter(csv_path)

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

            results = model.predict(detection_frame, conf=conf, iou=iou, imgsz=img_size, verbose=False)

            raw_detections = []
            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        score = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        raw_detections.append(
                            [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), score, cls_id]
                        )

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

            csv_writer.write_frame(processed_count, timestamp, tracked_objects)
            video_writer.write(annotated)

            # Save preview for EVERY processed frame
            preview_name = f"preview_{processed_count:06d}.jpg"
            preview_path = previews_dir / preview_name
            cv2.imwrite(str(preview_path), annotated)
            preview_count += 1

            processed_count += 1
            frame_idx += 1
            elapsed = time.perf_counter() - t_start
            with job.lock:
                job.frame_index = frame_idx
                job.processed_frames = processed_count
                job.active_tracks = len(tracked_objects)
                job.unique_tracks = tracker.total_tracks_created
                job.csv_rows = csv_writer.rows_written if csv_writer is not None else 0
                job.progress = min(frame_idx / total_frames, 1.0) if total_frames > 0 else 0.0
                job.elapsed_sec = elapsed
                job.previews_count = preview_count
                job.latest_preview_name = preview_name
                job.message = "Processing video..."

        if csv_writer is not None:
            csv_writer.flush()

        unique_tracks = _count_unique_tracks(csv_path)
        csv_rows = csv_writer.rows_written if csv_writer is not None else 0
        elapsed = time.perf_counter() - t_start
        with job.lock:
            job.status = "done"
            job.progress = 1.0
            job.unique_tracks = unique_tracks
            job.csv_rows = csv_rows
            job.elapsed_sec = elapsed
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
        if csv_writer is not None:
            csv_writer.close()
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
    roi_points: str = Form("[]"),
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

    upload_dir = video_path.parent

    job_id = uuid.uuid4().hex[:12]
    job = JobState(job_id=job_id, upload_dir=upload_dir)
    jobs[job_id] = job

    thread = threading.Thread(
        target=_run_job,
        kwargs={
            "job": job,
            "video_path": video_path,
            "model_path": model,
            "interval": max(1, _parse_int(interval, 1)),
            "roi_points": roi if roi else None,
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
            "latest_preview_url": (
                f"/api/jobs/{job_id}/preview/{job.latest_preview_name}"
                if job.latest_preview_name
                else None
            ),
            "csv_download_url": f"/api/jobs/{job_id}/download/csv" if job.status == "done" else None,
            "video_download_url": f"/api/jobs/{job_id}/download/video" if job.status == "done" else None,
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
