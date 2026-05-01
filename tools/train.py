"""
ViTTA — Training Launcher

Launches YOLOv8 fine-tuning on the merged dataset with optimized
hyperparameters for Indian traffic detection.

Features:
  - Auto-detects GPU (RTX A5000) and uses it
  - Resumes from IDD_YOLO/best.pt (transfer learning)
  - Optimized augmentation for dense Indian traffic
  - Early stopping with patience
  - Versioned output directories

Usage:
    python train.py
    python train.py --data datasets/merged/data.yaml --epochs 50
    python train.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vitta.train")


def main():
    parser = argparse.ArgumentParser(
        description="ViTTA — YOLOv8 Training Launcher (optimized for Indian traffic)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data", type=str, default="datasets/merged/data.yaml",
        help="Path to the YOLO data.yaml for training.",
    )
    parser.add_argument(
        "--model", type=str, default="IDD_YOLO/best.pt",
        help="Base model to fine-tune from.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size (16 fits comfortably in 24GB VRAM at imgsz=720).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=720,
        help="Training image size.",
    )
    parser.add_argument(
        "--name", type=str, default="vitta_merged",
        help="Run name for the output directory.",
    )
    parser.add_argument(
        "--patience", type=int, default=15,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last checkpoint.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of data loading workers.",
    )
    args = parser.parse_args()

    # Validate paths
    data_path = Path(args.data)
    model_path = Path(args.model)

    if not data_path.exists():
        logger.error(f"Data config not found: {data_path}")
        logger.error("Run the conversion and merge scripts first:")
        logger.error("  python convert_uvh26.py")
        logger.error("  python convert_idd.py")
        logger.error("  python merge_datasets.py")
        sys.exit(1)

    if not model_path.exists():
        logger.error(f"Base model not found: {model_path}")
        sys.exit(1)

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU detected: {gpu_name} ({vram:.0f}GB VRAM)")
            device = "0"
        else:
            logger.warning("No CUDA GPU detected! Training on CPU will be very slow.")
            device = "cpu"
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        device = "cpu"

    # Load YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("ViTTA Training Configuration")
    logger.info("=" * 60)
    logger.info(f"  Base model:    {model_path}")
    logger.info(f"  Dataset:       {data_path}")
    logger.info(f"  Device:        {device}")
    logger.info(f"  Epochs:        {args.epochs}")
    logger.info(f"  Batch size:    {args.batch}")
    logger.info(f"  Image size:    {args.imgsz}")
    logger.info(f"  Patience:      {args.patience}")
    logger.info(f"  Workers:       {args.workers}")
    logger.info("=" * 60)

    # Load model
    model = YOLO(str(model_path))

    # Train with optimized hyperparameters for Indian traffic
    results = model.train(
        data=str(data_path.resolve()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project="runs/train",
        name=args.name,
        exist_ok=True,
        patience=args.patience,
        resume=args.resume,

        # ── Optimizer ─────────────────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,          # Lower LR since we're fine-tuning, not from scratch
        lrf=0.01,           # Final LR = lr0 * lrf
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # ── Augmentation (optimized for dense Indian traffic) ─────
        mosaic=1.0,         # Heavy mosaic — teaches model to handle occlusion
        mixup=0.15,         # Moderate mixup — helps with overlapping vehicles
        degrees=5.0,        # Slight rotation for camera angle variance
        translate=0.1,
        scale=0.5,
        flipud=0.0,         # No vertical flip (vehicles don't appear upside down)
        fliplr=0.5,         # Horizontal flip is fine
        hsv_h=0.015,        # Color jitter for varying lighting
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,        # Random erasing — helps with partial occlusion
        close_mosaic=10,    # Disable mosaic for last 10 epochs (fine-tuning)

        # ── Other ─────────────────────────────────────────────────
        amp=True,           # Mixed precision for speed
        deterministic=False, # Faster training
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"  Best model saved to: runs/train/{args.name}/weights/best.pt")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Evaluate: Check runs/train/{}/results.png".format(args.name))
    logger.info("  2. Test: python main.py track --video <video> --model runs/train/{}/weights/best.pt".format(args.name))
    logger.info("  3. Deploy: Copy best.pt to IDD_YOLO/best.pt to replace the old model")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
