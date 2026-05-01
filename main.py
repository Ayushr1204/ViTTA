"""
ViTTA — Video-based Traffic Trajectory extraction and Analysis

Unified CLI with two subcommands:

    python main.py extract  --video <path>                 # frame extraction only
    python main.py track    --video <path> --model <path>  # full tracking pipeline
    python main.py --help
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vitta")


# ═══════════════════════════════════════════════════════════════════════
#  Shared helpers
# ═══════════════════════════════════════════════════════════════════════

def _validate_video(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        logger.error(f"Video not found: {p}")
        sys.exit(1)
    return p


# ═══════════════════════════════════════════════════════════════════════
#  Subcommand: extract
# ═══════════════════════════════════════════════════════════════════════

def _add_extract_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="frames",
        help="Output directory for extracted frames.",
    )
    parser.add_argument(
        "--interval", "-i", type=int, default=3,
        help="Extract every Nth frame (default: 3 → 10 fps from 30 fps source).",
    )
    parser.add_argument(
        "--roi", action="store_true",
        help="Interactively select a Region of Interest on the first frame.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Run the pipeline without saving frames (dry-run / benchmark).",
    )
    parser.add_argument("--no-contrast", action="store_true", help="Skip CLAHE.")
    parser.add_argument("--no-denoise", action="store_true", help="Skip denoising.")
    parser.add_argument("--no-sharpen", action="store_true", help="Skip sharpening.")


def run_extract(args: argparse.Namespace) -> None:
    """Execute the frame extraction → preprocessing → save pipeline."""
    from vitta.config import PipelineConfig
    from vitta.frame_extractor import FrameExtractor
    from vitta.preprocessor import FramePreprocessor
    from vitta.roi_selector import select_roi_from_frame

    video_path = _validate_video(args.video)

    config = PipelineConfig(
        frame_sample_interval=args.interval,
        output_dir=Path(args.output),
        save_preprocessed_frames=not args.no_save,
        apply_denoise=not args.no_denoise,
        apply_sharpening=not args.no_sharpen,
    )
    if args.no_contrast:
        config.clahe_clip_limit = 1.0

    extractor = FrameExtractor(video_path, sample_interval=config.frame_sample_interval)

    # Optional interactive ROI selection
    if args.roi:
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        cap.release()
        if ret:
            roi = select_roi_from_frame(first_frame)
            if roi is not None:
                config.roi = roi
        else:
            logger.warning("Could not read first frame for ROI selection.")

    preprocessor = FramePreprocessor(config)

    extracted_dir = config.output_dir / "extracted"
    if config.save_preprocessed_frames:
        extracted_dir.mkdir(parents=True, exist_ok=True)

    video_info = extractor.get_video_info()
    logger.info("=" * 60)
    logger.info("Starting frame extraction and preprocessing pipeline")
    logger.info("=" * 60)
    for key, val in video_info.items():
        logger.info(f"  {key}: {val}")
    logger.info("=" * 60)

    t_start = time.perf_counter()
    frame_count = 0

    for frame_data in extractor.extract():
        processed = preprocessor.process(frame_data.image)

        if config.save_preprocessed_frames:
            filename = f"frame_{frame_data.sampled_index:06d}_t{frame_data.timestamp_sec:.2f}s.jpg"
            cv2.imwrite(str(extracted_dir / filename), processed)

        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.perf_counter() - t_start
            fps_actual = frame_count / elapsed if elapsed > 0 else 0
            logger.info(
                f"  Processed {frame_count}/{extractor.expected_sampled_frames} frames "
                f"({fps_actual:.1f} frames/sec)"
            )

    elapsed = time.perf_counter() - t_start
    fps_actual = frame_count / elapsed if elapsed > 0 else 0
    extractor.release()

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"  Frames extracted:  {frame_count}")
    logger.info(f"  Total time:        {elapsed:.2f}s")
    logger.info(f"  Processing speed:  {fps_actual:.1f} frames/sec")
    if config.save_preprocessed_frames:
        logger.info(f"  Extracted frames:  {extracted_dir}")
    logger.info("=" * 60)

    # Save metadata JSON
    meta = {
        **video_info,
        "frames_extracted": frame_count,
        "processing_time_sec": round(elapsed, 2),
        "config": {
            "roi": config.roi,
            "clahe_clip_limit": config.clahe_clip_limit,
            "clahe_tile_grid_size": config.clahe_tile_grid_size,
            "apply_denoise": config.apply_denoise,
            "apply_sharpening": config.apply_sharpening,
            "target_resolution": config.target_resolution,
        },
    }
    meta_path = config.output_dir / "extraction_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Metadata saved:    {meta_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Subcommand: track
# ═══════════════════════════════════════════════════════════════════════

def _add_track_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Path to YOLO weights (.pt file).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output",
        help="Output directory for tracked video, CSV, and Excel.",
    )

    # Output toggles
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV output.")
    parser.add_argument("--no-video", action="store_true", help="Skip annotated video.")
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel export.")

    # Display
    parser.add_argument(
        "--show", action="store_true",
        help="Display live tracking window (press 'q' to quit).",
    )

    # Detection
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO NMS IoU threshold.")
    parser.add_argument("--img-size", type=int, default=640, help="YOLO input image size.")

    # Tracker
    parser.add_argument("--track-high", type=float, default=0.6, help="ByteTrack high threshold.")
    parser.add_argument("--track-low", type=float, default=0.1, help="ByteTrack low threshold.")
    parser.add_argument("--track-buffer", type=int, default=60, help="Lost-track buffer (frames).")
    parser.add_argument("--new-track-thresh", type=float, default=0.7, help="New track threshold.")

    # Sampling
    parser.add_argument("--interval", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument(
        "--resample-interval", type=float, default=1.0,
        help="Resample interval in seconds for the final dataset.",
    )


def run_track(args: argparse.Namespace) -> None:
    """Run the full detection → tracking → resampling → export pipeline."""
    from vitta.tracking.config import TrackerConfig
    from vitta.tracking.tracker import ByteTracker
    from vitta.tracking.visualizer import TrackVisualizer
    from vitta.tracking.csv_writer import TrackCSVWriter
    from vitta.tracking.resampler import TrackResampler
    from vitta.export.excel_writer import ExcelExporter

    video_path = _validate_video(args.video)
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    logger.info(f"Loading YOLO model: {model_path}")
    model = YOLO(str(model_path))

    # Open video
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

    # Initialise components
    tracker_config = TrackerConfig(
        track_high_thresh=args.track_high,
        track_low_thresh=args.track_low,
        new_track_thresh=args.new_track_thresh,
        track_buffer=args.track_buffer,
        csv_output_path=output_dir / "tracks.csv",
    )
    tracker = ByteTracker(tracker_config)
    visualizer = TrackVisualizer(tracker_config)
    resampler = TrackResampler(interval_sec=args.resample_interval)

    video_writer = None
    if not args.no_video:
        out_video_path = output_dir / "tracked_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(out_video_path), fourcc, effective_fps, (width, height)
        )
        logger.info(f"Output video: {out_video_path}")

    csv_writer = None
    if not args.no_csv:
        csv_writer = TrackCSVWriter(output_dir / "tracks.csv")

    # Processing loop
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

            if frame_idx % args.interval != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps

            # YOLO detection
            results = model.predict(
                frame, conf=args.conf, iou=args.iou,
                imgsz=args.img_size, verbose=False,
            )

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
                            float(xyxy[0]), float(xyxy[1]),
                            float(xyxy[2]), float(xyxy[3]),
                            conf, cls,
                        ])

            # Track
            tracked_objects = tracker.update(raw_detections, processed_count)

            # Feed resampler
            resampler.add_from_tracked_objects(
                frame_id=processed_count,
                timestamp=timestamp,
                tracked_objects=tracked_objects,
            )

            # Visualise
            annotated = visualizer.draw(frame, tracked_objects)

            if video_writer is not None:
                video_writer.write(annotated)
            if csv_writer is not None:
                csv_writer.write_frame(processed_count, timestamp, tracked_objects)

            if args.show:
                cv2.imshow("ViTTA Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User pressed 'q' — stopping.")
                    break

            processed_count += 1
            frame_idx += 1

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
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if csv_writer is not None:
            csv_writer.close()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    speed = processed_count / elapsed if elapsed > 0 else 0

    logger.info("=" * 60)
    logger.info("Tracking complete!")
    logger.info(f"  Frames processed:      {processed_count}")
    logger.info(f"  Total time:            {elapsed:.2f}s")
    logger.info(f"  Processing speed:      {speed:.1f} fps")
    logger.info(f"  Unique tracks created: {tracker.total_tracks_created}")
    logger.info(f"  Active tracks (end):   {len(tracker.get_active_tracks())}")
    if csv_writer is not None:
        logger.info(f"  CSV rows written:      {csv_writer.rows_written}")
        logger.info(f"  CSV path:              {output_dir / 'tracks.csv'}")
    if not args.no_video:
        logger.info(f"  Output video:          {output_dir / 'tracked_output.mp4'}")
    logger.info("=" * 60)

    # Post-processing: resample + Excel
    logger.info("")
    logger.info("=" * 60)
    logger.info("Post-processing: resampling to 1-second intervals...")
    logger.info("=" * 60)

    resampled_records = resampler.resample()

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
        logger.info(f"  Excel dataset:         {excel_path}")
    elif not resampled_records:
        logger.warning("  No resampled records to export (no tracks found).")

    logger.info("=" * 60)
    logger.info("All done!")
    logger.info("=" * 60)


# ═══════════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog="vitta",
        description="ViTTA — Video-based Traffic Trajectory extraction and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python main.py extract --video traffic.mp4
  python main.py track   --video traffic.mp4 --model IDD_YOLO/best.pt
  python main.py track   --video traffic.mp4 --model IDD_YOLO/best.pt --show
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract subcommand
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract and preprocess frames from a traffic video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_extract_args(extract_parser)

    # track subcommand
    track_parser = subparsers.add_parser(
        "track",
        help="Detect, track, and analyse road users (full pipeline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_track_args(track_parser)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)
    elif args.command == "extract":
        run_extract(args)
    elif args.command == "track":
        run_track(args)


if __name__ == "__main__":
    main()
