"""
ViTTA — Track Demo

End-to-end integration script that reads a video, runs YOLO detection,
feeds detections to ByteTracker, visualises the results, and saves
both an annotated video and a CSV log.

Post-processing: resamples tracks to 1-second intervals and exports
a structured Excel workbook.

Usage:
    python track_demo.py --video path/to/video.mp4 --model path/to/yolo.pt
    python track_demo.py --video traffic.mp4 --model best.pt --show
    python track_demo.py --video traffic.mp4 --model best.pt --no-csv
    python track_demo.py --help
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from vitta.tracking.config import TrackerConfig
from vitta.tracking.tracker import ByteTracker
from vitta.tracking.visualizer import TrackVisualizer
from vitta.tracking.csv_writer import TrackCSVWriter
from vitta.tracking.resampler import TrackResampler
from vitta.export.excel_writer import ExcelExporter

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vitta.track_demo")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViTTA — ByteTrack vehicle tracking demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────
    p.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    p.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to YOLO weights (.pt file).",
    )

    # ── Output ────────────────────────────────────────────────────────
    p.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory for tracked video and CSV.",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV output.",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing the annotated output video.",
    )
    p.add_argument(
        "--no-excel",
        action="store_true",
        help="Skip Excel (.xlsx) export.",
    )

    # ── Display ───────────────────────────────────────────────────────
    p.add_argument(
        "--show",
        action="store_true",
        help="Display live tracking window (press 'q' to quit).",
    )

    # ── Detection ─────────────────────────────────────────────────────
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold.",
    )
    p.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="YOLO input image size.",
    )

    # ── Tracker overrides ─────────────────────────────────────────────
    p.add_argument(
        "--track-high",
        type=float,
        default=0.6,
        help="ByteTrack high-confidence threshold.",
    )
    p.add_argument(
        "--track-low",
        type=float,
        default=0.1,
        help="ByteTrack low-confidence threshold.",
    )
    p.add_argument(
        "--track-buffer",
        type=int,
        default=60,
        help="Frames to keep lost tracks alive.",
    )
    p.add_argument(
        "--new-track-thresh",
        type=float,
        default=0.7,
        help="Minimum confidence to create a new track.",
    )

    # ── Frame sampling ────────────────────────────────────────────────
    p.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Process every Nth frame (1 = every frame).",
    )

    # ── Resampling ────────────────────────────────────────────────────
    p.add_argument(
        "--resample-interval",
        type=float,
        default=1.0,
        help="Resample interval in seconds for the final dataset.",
    )

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    """Run the full detection → tracking → visualisation pipeline."""

    # ── Validate inputs ───────────────────────────────────────────────
    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load YOLO model ───────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error(
            "ultralytics is not installed. Run:  pip install ultralytics"
        )
        sys.exit(1)

    logger.info(f"Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    effective_fps = fps / args.interval

    logger.info(
        f"Video: {video_path.name} | {width}×{height} @ {fps:.1f}fps | "
        f"{total_frames} frames | processing every {args.interval} frame(s)"
    )

    # ── Initialise tracker ────────────────────────────────────────────
    tracker_config = TrackerConfig(
        track_high_thresh=args.track_high,
        track_low_thresh=args.track_low,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        csv_output_path=output_dir / "tracks.csv",
    )
    tracker = ByteTracker(tracker_config)
    visualizer = TrackVisualizer(tracker_config)

    # ── Initialise output video writer ────────────────────────────────
    video_writer = None
    if not args.no_video:
        out_video_path = output_dir / "tracked_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(out_video_path), fourcc, effective_fps, (width, height)
        )
        logger.info(f"Output video: {out_video_path}")

    # ── Initialise CSV writer ─────────────────────────────────────────
    csv_writer = None
    if not args.no_csv:
        csv_writer = TrackCSVWriter(output_dir / "tracks.csv")

    # ── Initialise resampler ──────────────────────────────────────────
    resampler = TrackResampler(interval_sec=args.resample_interval)

    # ── Processing loop ───────────────────────────────────────────────
    frame_idx = 0
    processed_count = 0
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Starting tracking pipeline...")
    logger.info("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames based on interval
            if frame_idx % args.interval != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps

            # ── YOLO detection ────────────────────────────────────────
            results = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.img_size,
                verbose=False,
            )

            # Convert YOLO results → [x1, y1, x2, y2, confidence, class_id]
            raw_detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    for i in range(len(boxes)):
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())
                        raw_detections.append([
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                            conf,
                            cls,
                        ])

            # ── Track ─────────────────────────────────────────────────
            tracked_objects = tracker.update(raw_detections, processed_count)

            # ── Feed resampler ────────────────────────────────────────
            resampler.add_from_tracked_objects(
                frame_id=processed_count,
                timestamp=timestamp,
                tracked_objects=tracked_objects,
            )

            # ── Visualise ─────────────────────────────────────────────
            annotated = visualizer.draw(frame, tracked_objects)

            # ── Write outputs ─────────────────────────────────────────
            if video_writer is not None:
                video_writer.write(annotated)

            if csv_writer is not None:
                csv_writer.write_frame(processed_count, timestamp, tracked_objects)

            # ── Live display ──────────────────────────────────────────
            if args.show:
                cv2.imshow("ViTTA Tracking", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User pressed 'q' — stopping.")
                    break

            processed_count += 1
            frame_idx += 1

            # Progress log every 100 processed frames
            if processed_count % 100 == 0:
                elapsed = time.perf_counter() - t_start
                speed = processed_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  Frame {processed_count} | "
                    f"{len(tracked_objects)} active tracks | "
                    f"{speed:.1f} fps"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        # ── Cleanup ───────────────────────────────────────────────────
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if csv_writer is not None:
            csv_writer.close()
        if args.show:
            cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    speed = processed_count / elapsed if elapsed > 0 else 0

    logger.info("=" * 60)
    logger.info("Tracking complete!")
    logger.info(f"  Frames processed:     {processed_count}")
    logger.info(f"  Total time:           {elapsed:.2f}s")
    logger.info(f"  Processing speed:     {speed:.1f} fps")
    logger.info(f"  Unique tracks created: {tracker.total_tracks_created}")
    logger.info(f"  Active tracks (end):   {len(tracker.get_active_tracks())}")
    if csv_writer is not None:
        logger.info(f"  CSV rows written:     {csv_writer.rows_written}")
        logger.info(f"  CSV path:             {output_dir / 'tracks.csv'}")
    if not args.no_video:
        logger.info(f"  Output video:         {output_dir / 'tracked_output.mp4'}")
    logger.info("=" * 60)

    # ── Post-processing: resample to 1-second intervals ───────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("Post-processing: resampling to 1-second intervals...")
    logger.info("=" * 60)

    resampled_records = resampler.resample()

    # ── Excel export ──────────────────────────────────────────────────
    if not args.no_excel and resampled_records:
        excel_path = output_dir / "vitta_results.xlsx"
        metadata = {
            "video_file": str(video_path),
            "video_resolution": f"{width}x{height}",
            "video_fps": fps,
            "total_frames": total_frames,
            "frames_processed": processed_count,
            "processing_interval": args.interval,
            "effective_fps": effective_fps,
            "resample_interval_sec": args.resample_interval,
            "yolo_model": str(model_path),
            "yolo_conf_threshold": args.conf,
            "yolo_iou_threshold": args.iou,
            "yolo_img_size": args.img_size,
            "tracker_high_thresh": args.track_high,
            "tracker_low_thresh": args.track_low,
            "tracker_new_track_thresh": args.new_track_thresh,
            "tracker_buffer_frames": args.track_buffer,
            "unique_tracks": tracker.total_tracks_created,
            "processing_time_sec": round(elapsed, 2),
            "processing_speed_fps": round(speed, 1),
        }

        exporter = ExcelExporter()
        exporter.export(resampled_records, excel_path, metadata)
        logger.info(f"  Excel dataset:        {excel_path}")
    elif not resampled_records:
        logger.warning("  No resampled records to export (no tracks found).")

    logger.info("=" * 60)
    logger.info("All done!")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
