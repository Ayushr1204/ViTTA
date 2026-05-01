"""
ViTTA — Dataset Merger (Symlink-based)

Merges multiple YOLO-format datasets into a single unified training
dataset using **symbolic links** instead of copying files. This saves
disk space by avoiding duplicate images.

Usage:
    python merge_datasets.py
    python merge_datasets.py --sources datasets/uvh26_yolo datasets/idd_yolo
    python merge_datasets.py --stats-only
    python merge_datasets.py --help
"""

import argparse
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("merge_datasets")

VITTA_CLASS_NAMES = {
    0: "Car", 1: "Bus", 2: "Truck", 3: "Auto",
    4: "2W", 5: "LCV", 6: "Bicycle", 7: "Pedestrian",
}


def count_labels(label_dir: Path) -> dict:
    """Count annotations per class in a YOLO label directory."""
    stats = defaultdict(int)
    for txt in label_dir.glob("*.txt"):
        for line in txt.read_text().strip().split("\n"):
            if line.strip():
                try:
                    cls_id = int(line.strip().split()[0])
                    stats[cls_id] += 1
                except ValueError:
                    pass
    return dict(stats)


def link_split(
    src_dir: Path,
    dst_img_dir: Path,
    dst_lbl_dir: Path,
    split: str,
) -> int:
    """Create symlinks from source dataset split into merged output."""
    src_imgs = src_dir / "images" / split
    src_lbls = src_dir / "labels" / split
    linked = 0

    if not src_imgs.exists():
        return 0

    for img_file in src_imgs.iterdir():
        if not img_file.is_file():
            continue

        # Create symlink for image
        dst_img = dst_img_dir / img_file.name
        if not dst_img.exists():
            try:
                os.symlink(img_file.resolve(), dst_img)
            except OSError:
                # Fallback: create a hardlink if symlinks need admin
                try:
                    os.link(img_file.resolve(), dst_img)
                except OSError:
                    # Last resort: just skip (will be handled below)
                    logger.warning(f"  Cannot link {img_file.name}, skipping")
                    continue

        # Create symlink for label
        lbl_name = img_file.stem + ".txt"
        src_lbl = src_lbls / lbl_name
        if src_lbl.exists():
            dst_lbl = dst_lbl_dir / lbl_name
            if not dst_lbl.exists():
                try:
                    os.symlink(src_lbl.resolve(), dst_lbl)
                except OSError:
                    try:
                        os.link(src_lbl.resolve(), dst_lbl)
                    except OSError:
                        pass

        linked += 1

    return linked


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO-format datasets using symlinks (saves disk space)",
    )
    parser.add_argument(
        "--sources", nargs="+", type=str,
        default=["datasets/uvh26_yolo", "datasets/idd_yolo"],
        help="Paths to YOLO-format dataset directories to merge.",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="datasets/merged",
        help="Output directory for the merged dataset.",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Only print statistics, don't merge.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    sources = [Path(s) for s in args.sources]

    logger.info("=" * 60)
    logger.info("ViTTA Dataset Merger (symlink-based)")
    logger.info("=" * 60)

    # Validate sources
    valid_sources = []
    for src in sources:
        if src.exists():
            valid_sources.append(src)
            logger.info(f"  Source: {src}")
        else:
            logger.warning(f"  Source not found (skipping): {src}")

    if not valid_sources:
        logger.error("No valid source datasets found!")
        return

    if args.stats_only:
        for src in valid_sources:
            logger.info(f"\n--- {src.name} ---")
            for split in ["train", "val"]:
                lbl_dir = src / "labels" / split
                if lbl_dir.exists():
                    stats = count_labels(lbl_dir)
                    img_dir = src / "images" / split
                    img_count = len(list(img_dir.glob("*"))) if img_dir.exists() else 0
                    logger.info(f"  {split}: {img_count} images")
                    for cls_id, count in sorted(stats.items()):
                        name = VITTA_CLASS_NAMES.get(cls_id, f"cls{cls_id}")
                        logger.info(f"    {cls_id}: {name:12s} = {count:,}")
        return

    t_start = time.perf_counter()

    # Create output directories
    train_img_dir = out_dir / "images" / "train"
    train_lbl_dir = out_dir / "labels" / "train"
    val_img_dir = out_dir / "images" / "val"
    val_lbl_dir = out_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Merge each source via symlinks
    total_train = 0
    total_val = 0

    for src in valid_sources:
        logger.info(f"\nLinking: {src.name}")

        n_train = link_split(src, train_img_dir, train_lbl_dir, "train")
        n_val = link_split(src, val_img_dir, val_lbl_dir, "val")

        logger.info(f"  Linked: {n_train} train, {n_val} val")
        total_train += n_train
        total_val += n_val

    elapsed = time.perf_counter() - t_start

    # Write data.yaml
    yaml_content = f"""# ViTTA Merged Training Dataset
# Sources: {', '.join(s.name for s in valid_sources)}
# Generated by merge_datasets.py (symlink-based, no disk duplication)

path: {out_dir.resolve()}
train: images/train
val: images/val

nc: 8
names: {list(VITTA_CLASS_NAMES.values())}
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Final statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("Merge complete!")
    logger.info(f"  Time:         {elapsed:.1f}s")
    logger.info(f"  Total train:  {total_train:,}")
    logger.info(f"  Total val:    {total_val:,}")
    logger.info(f"  Output:       {out_dir.resolve()}")
    logger.info(f"  data.yaml:    {yaml_path}")

    # Class distribution
    logger.info("")
    logger.info("Final class distribution (TRAIN):")
    train_stats = count_labels(train_lbl_dir)
    grand_total = 0
    for cls_id in range(8):
        count = train_stats.get(cls_id, 0)
        grand_total += count
        logger.info(f"  {cls_id}: {VITTA_CLASS_NAMES[cls_id]:12s} = {count:>8,}")
    logger.info(f"  {'TOTAL':>15s} = {grand_total:>8,}")

    logger.info("")
    logger.info("Final class distribution (VAL):")
    val_stats = count_labels(val_lbl_dir)
    grand_total = 0
    for cls_id in range(8):
        count = val_stats.get(cls_id, 0)
        grand_total += count
        logger.info(f"  {cls_id}: {VITTA_CLASS_NAMES[cls_id]:12s} = {count:>8,}")
    logger.info(f"  {'TOTAL':>15s} = {grand_total:>8,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
