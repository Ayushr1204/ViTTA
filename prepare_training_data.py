"""
ViTTA — Training Data Preparation Tool

Extracts diverse frames from traffic videos and generates YOLO-format
pseudo-labels using the current model, ready for manual correction
in annotation tools (CVAT, Roboflow, Label Studio, etc.).

Workflow:
  1. Reads a traffic video and samples frames at configurable intervals
  2. Runs the current YOLO model to generate initial bounding-box labels
  3. Saves frames as images + YOLO-format .txt annotation files
  4. Outputs a data.yaml for YOLO training

Usage:
    python prepare_training_data.py --video traffic.mp4 --model IDD_YOLO/best.pt
    python prepare_training_data.py --video traffic.mp4 --model IDD_YOLO/best.pt --interval 30 --conf 0.2
    python prepare_training_data.py --help
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import cv2

from vitta.class_names import CLASS_NAMES

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vitta.prepare_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ViTTA — Prepare YOLO training data from traffic videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video", "-v", type=str, required=True,
        help="Path to the input video file.",
    )
    p.add_argument(
        "--model", "-m", type=str, required=True,
        help="Path to YOLO weights for pseudo-labelling.",
    )
    p.add_argument(
        "--output", "-o", type=str, default="training_data",
        help="Output directory for the YOLO dataset.",
    )
    p.add_argument(
        "--interval", "-i", type=int, default=30,
        help="Extract every Nth frame (e.g. 30 = 1 frame/sec at 30fps).",
    )
    p.add_argument(
        "--max-frames", type=int, default=500,
        help="Maximum number of frames to extract.",
    )
    p.add_argument(
        "--conf", type=float, default=0.20,
        help="Confidence threshold for pseudo-labels (lower = more labels to review).",
    )
    p.add_argument(
        "--img-size", type=int, default=640,
        help="YOLO inference image size.",
    )
    p.add_argument(
        "--val-split", type=float, default=0.15,
        help="Fraction of frames to put in the validation set.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split.",
    )
    return p.parse_args()


def _xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float,
                  img_w: int, img_h: int) -> tuple:
    """Convert [x1,y1,x2,y2] pixel coords to YOLO format [cx,cy,w,h] normalised."""
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    # Clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return cx, cy, w, h


def run(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    # ── Setup output directory structure ──────────────────────────────
    out_dir = Path(args.output)
    train_img_dir = out_dir / "images" / "train"
    train_lbl_dir = out_dir / "labels" / "train"
    val_img_dir = out_dir / "images" / "val"
    val_lbl_dir = out_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Load YOLO ─────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info(f"Loading model: {model_path}")
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

    logger.info(f"Video: {video_path.name} | {width}x{height} @ {fps:.0f}fps | {total_frames} frames")
    logger.info(f"Sampling every {args.interval} frames → ~{min(total_frames // args.interval, args.max_frames)} frames")

    # ── Extract frames & pseudo-label ────────────────────────────────
    frame_idx = 0
    extracted = 0
    frame_names = []
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Extracting frames and generating pseudo-labels...")
    logger.info("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret or extracted >= args.max_frames:
            break

        if frame_idx % args.interval != 0:
            frame_idx += 1
            continue

        # Run detection
        results = model.predict(
            frame, conf=args.conf, imgsz=args.img_size, verbose=False,
        )

        # Build YOLO annotation lines
        lines = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                cx, cy, w, h = _xyxy_to_yolo(
                    float(xyxy[0]), float(xyxy[1]),
                    float(xyxy[2]), float(xyxy[3]),
                    width, height,
                )
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save image and label (initially all to train; split later)
        fname = f"{video_path.stem}_{frame_idx:06d}"
        frame_names.append(fname)

        cv2.imwrite(str(train_img_dir / f"{fname}.jpg"), frame)
        with open(train_lbl_dir / f"{fname}.txt", "w") as f:
            f.write("\n".join(lines))

        extracted += 1
        frame_idx += 1

        if extracted % 50 == 0:
            logger.info(f"  Extracted {extracted} frames ({len(lines)} detections in last frame)")

    cap.release()
    elapsed = time.perf_counter() - t_start
    logger.info(f"Extracted {extracted} frames in {elapsed:.1f}s")

    # ── Train / val split ────────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(frame_names)
    val_count = max(1, int(len(frame_names) * args.val_split))
    val_names = set(frame_names[:val_count])

    moved = 0
    for fname in val_names:
        # Move image
        src_img = train_img_dir / f"{fname}.jpg"
        dst_img = val_img_dir / f"{fname}.jpg"
        if src_img.exists():
            src_img.rename(dst_img)
        # Move label
        src_lbl = train_lbl_dir / f"{fname}.txt"
        dst_lbl = val_lbl_dir / f"{fname}.txt"
        if src_lbl.exists():
            src_lbl.rename(dst_lbl)
        moved += 1

    train_count = extracted - moved
    logger.info(f"Split: {train_count} train / {moved} val")

    # ── Write data.yaml ──────────────────────────────────────────────
    yaml_content = f"""# ViTTA YOLO Training Dataset
# Auto-generated pseudo-labels — REVIEW AND CORRECT before training!
# Source video: {video_path.name}

path: {out_dir.resolve()}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names: {list(CLASS_NAMES.values())}
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Training data preparation complete!")
    logger.info(f"  Output directory: {out_dir.resolve()}")
    logger.info(f"  Train images:     {train_count}")
    logger.info(f"  Val images:       {moved}")
    logger.info(f"  data.yaml:        {yaml_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review and correct labels using an annotation tool:")
    logger.info("     - CVAT (https://www.cvat.ai/)")
    logger.info("     - Roboflow (https://roboflow.com/)")
    logger.info("     - Label Studio (https://labelstud.io/)")
    logger.info("  2. Fine-tune the model:")
    logger.info(f"     yolo detect train data={yaml_path} model=IDD_YOLO/best.pt epochs=30 imgsz=720")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run(args)
