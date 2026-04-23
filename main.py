"""
ViTTA - Video-based Traffic Trajectory extraction and Analysis

Main entry point for the frame extraction and preprocessing pipeline.

Usage:
    python main.py
    python main.py --roi
    python main.py --no-save
"""

# ── Change this to your video file path ───────────────────────────────
VIDEO_PATH = r"C:\path\to\your\traffic_video.mp4"

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from vitta.config import PipelineConfig
from vitta.frame_extractor import FrameExtractor
from vitta.preprocessor import FramePreprocessor
from vitta.roi_selector import select_roi_from_frame

# ── Logging setup ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vitta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViTTA - Extract and preprocess frames from traffic videos",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="frames",
        help="Output directory for extracted frames (default: ./frames)",
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=3,
        help="Extract every Nth frame (default: 3, giving 10 fps from a 30 fps source)",
    )
    parser.add_argument(
        "--roi",
        action="store_true",
        default=False,
        help="Interactively select a Region of Interest on the first frame",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Run the pipeline without saving frames to disk (dry-run / benchmark)",
    )
    parser.add_argument(
        "--no-contrast",
        action="store_true",
        default=False,
        help="Skip CLAHE contrast enhancement",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        default=False,
        help="Skip Gaussian denoising",
    )
    parser.add_argument(
        "--no-sharpen",
        action="store_true",
        default=False,
        help="Skip sharpening filter",
    )
    return parser.parse_args()


def get_video_path() -> Path:
    """Resolve video path from VIDEO_PATH constant."""
    p = Path(VIDEO_PATH)
    if not p.exists():
        logger.error(f"File not found: {p}")
        sys.exit(1)
    return p


def run_pipeline(video_path: Path, args: argparse.Namespace) -> None:
    """Execute the frame extraction → preprocessing → save pipeline."""

    # ── Build configuration ───────────────────────────────────────────
    config = PipelineConfig(
        frame_sample_interval=args.interval,
        output_dir=Path(args.output),
        save_preprocessed_frames=not args.no_save,
        apply_denoise=not args.no_denoise,
        apply_sharpening=not args.no_sharpen,
    )

    # If contrast enhancement is disabled, use a neutral CLAHE (clip=1.0)
    if args.no_contrast:
        config.clahe_clip_limit = 1.0

    # ── Frame extractor ───────────────────────────────────────────────
    extractor = FrameExtractor(video_path, sample_interval=config.frame_sample_interval)

    # ── Optional interactive ROI selection ────────────────────────────
    if args.roi:
        # Read the first frame for ROI selection
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        cap.release()

        if ret:
            roi = select_roi_from_frame(first_frame)
            if roi is not None:
                config.roi = roi
        else:
            logger.warning("Could not read first frame for ROI selection, proceeding with full frame.")

    # ── Preprocessor ──────────────────────────────────────────────────
    preprocessor = FramePreprocessor(config)

    # ── Prepare output directories ────────────────────────────────────
    extracted_dir = config.output_dir / "extracted"

    if config.save_preprocessed_frames:
        extracted_dir.mkdir(parents=True, exist_ok=True)

    # ── Process frames ────────────────────────────────────────────────
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
        # Preprocess
        processed = preprocessor.process(frame_data.image)

        # Save to disk
        if config.save_preprocessed_frames:
            filename = f"frame_{frame_data.sampled_index:06d}_t{frame_data.timestamp_sec:.2f}s.jpg"
            cv2.imwrite(str(extracted_dir / filename), processed)

        frame_count += 1

        # Progress log every 100 frames
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

    # ── Summary ───────────────────────────────────────────────────────
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


def main():
    args = parse_args()
    video_path = get_video_path()
    run_pipeline(video_path, args)


if __name__ == "__main__":
    main()
