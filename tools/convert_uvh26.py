"""
ViTTA — UVH-26 Dataset Converter

Converts UVH-26 from COCO JSON → YOLO format, remapping the 14 UVH-26
vehicle classes to ViTTA's 8-class scheme.

UVH-26 classes (COCO, 1-indexed):
  1:Hatchback  2:Sedan  3:SUV  4:MUV  5:Bus  6:Truck  7:Three-wheeler
  8:Two-wheeler  9:LCV  10:Mini-bus  11:Tempo-traveller  12:Bicycle
  13:Van  14:Others

ViTTA classes (YOLO, 0-indexed):
  0:Car  1:Bus  2:Truck  3:Auto  4:2W  5:LCV  6:Bicycle  7:Pedestrian

Mapping:
  Hatchback, Sedan, SUV, MUV → 0 (Car)
  Bus, Mini-bus              → 1 (Bus)
  Truck                      → 2 (Truck)
  Three-wheeler              → 3 (Auto)
  Two-wheeler                → 4 (2W)
  LCV, Van, Tempo-traveller  → 5 (LCV)
  Bicycle                    → 6 (Bicycle)
  Others                     → SKIP (no ViTTA equivalent)

Usage:
    python convert_uvh26.py
    python convert_uvh26.py --annotation-variant ST  # use STAPLE labels
    python convert_uvh26.py --help
"""

import argparse
import json
import logging
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_uvh26")

# ── Class remapping ───────────────────────────────────────────────────
# UVH-26 COCO category_id → ViTTA YOLO class_id
UVH26_TO_VITTA = {
    1: 0,   # Hatchback   → Car
    2: 0,   # Sedan       → Car
    3: 0,   # SUV         → Car
    4: 0,   # MUV         → Car
    5: 1,   # Bus         → Bus
    6: 2,   # Truck       → Truck
    7: 3,   # Three-wheeler → Auto
    8: 4,   # Two-wheeler → 2W
    9: 5,   # LCV         → LCV
    10: 1,  # Mini-bus    → Bus
    11: 5,  # Tempo-traveller → LCV
    12: 6,  # Bicycle     → Bicycle
    13: 5,  # Van         → LCV
    14: None,  # Others   → SKIP
}

VITTA_CLASS_NAMES = {
    0: "Car", 1: "Bus", 2: "Truck", 3: "Auto",
    4: "2W", 5: "LCV", 6: "Bicycle", 7: "Pedestrian",
}


def convert_split(
    json_path: Path,
    image_dir: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    prefix: str,
) -> dict:
    """Convert one split (train or val) from COCO JSON to YOLO format."""
    logger.info(f"Loading annotations: {json_path.name}")
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Build image lookup: image_id → image info
    images = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skipped_others = 0
    converted_images = 0

    for img_id, img_info in images.items():
        filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Find the actual image file (may be in subfolders like 000/, 001/)
        src_img = image_dir / filename
        if not src_img.exists():
            # Try searching subdirectories
            candidates = list(image_dir.glob(f"**/{filename}"))
            if candidates:
                src_img = candidates[0]
            else:
                continue  # Image not found, skip

        # Convert annotations
        yolo_lines = []
        for ann in ann_by_image.get(img_id, []):
            cat_id = ann["category_id"]
            vitta_cls = UVH26_TO_VITTA.get(cat_id)

            if vitta_cls is None:
                skipped_others += 1
                continue

            # COCO bbox: [x_topleft, y_topleft, width, height]
            bx, by, bw, bh = ann["bbox"]
            # Convert to YOLO: [cx, cy, w, h] normalized
            cx = (bx + bw / 2.0) / img_w
            cy = (by + bh / 2.0) / img_h
            nw = bw / img_w
            nh = bh / img_h

            # Clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            # Skip degenerate boxes
            if nw < 0.001 or nh < 0.001:
                continue

            yolo_lines.append(f"{vitta_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[vitta_cls] += 1

        # Use a unique filename with prefix to avoid collisions
        stem = src_img.stem
        new_name = f"{prefix}_{stem}"

        # Copy image
        dst_img = out_img_dir / f"{new_name}{src_img.suffix}"
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write YOLO label
        with open(out_lbl_dir / f"{new_name}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

        converted_images += 1
        if converted_images % 2000 == 0:
            logger.info(f"  Converted {converted_images} images...")

    logger.info(f"  Done: {converted_images} images, {skipped_others} 'Others' annotations skipped")
    return dict(stats), converted_images


def main():
    parser = argparse.ArgumentParser(
        description="Convert UVH-26 (COCO JSON) → YOLO format with ViTTA class remapping",
    )
    parser.add_argument(
        "--dataset-dir", type=str,
        default="datasets/UVH-26-data",
        help="Path to the UVH-26 dataset root.",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="datasets/uvh26_yolo",
        help="Output directory for the YOLO-format dataset.",
    )
    parser.add_argument(
        "--annotation-variant", type=str, default="MV",
        choices=["MV", "ST"],
        help="Which annotation consensus to use: MV (majority voting) or ST (STAPLE).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output)
    variant = args.annotation_variant

    logger.info("=" * 60)
    logger.info(f"UVH-26 → YOLO Converter (using {variant} annotations)")
    logger.info("=" * 60)

    # Paths
    train_json = dataset_dir / "UVH-26-Train" / f"UVH-26-{variant}-Train.json"
    val_json = dataset_dir / "UVH-26-Val" / f"UVH-26-{variant}-Val.json"
    train_imgs = dataset_dir / "UVH-26-Train" / "data"
    val_imgs = dataset_dir / "UVH-26-Val" / "data"

    for p in [train_json, val_json, train_imgs, val_imgs]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            sys.exit(1)

    t_start = time.perf_counter()

    # Convert train split
    logger.info("Converting TRAIN split...")
    train_stats, train_count = convert_split(
        train_json, train_imgs,
        out_dir / "images" / "train",
        out_dir / "labels" / "train",
        prefix="uvh",
    )

    # Convert val split
    logger.info("Converting VAL split...")
    val_stats, val_count = convert_split(
        val_json, val_imgs,
        out_dir / "images" / "val",
        out_dir / "labels" / "val",
        prefix="uvh",
    )

    elapsed = time.perf_counter() - t_start

    # Write data.yaml
    yaml_content = f"""# UVH-26 converted to ViTTA 8-class YOLO format
# Annotation variant: {variant}

path: {out_dir.resolve()}
train: images/train
val: images/val

nc: 8
names: {list(VITTA_CLASS_NAMES.values())}
"""
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Summary
    logger.info("=" * 60)
    logger.info("Conversion complete!")
    logger.info(f"  Time:         {elapsed:.1f}s")
    logger.info(f"  Train images: {train_count}")
    logger.info(f"  Val images:   {val_count}")
    logger.info(f"  Output:       {out_dir.resolve()}")
    logger.info(f"  data.yaml:    {yaml_path}")
    logger.info("")
    logger.info("Class distribution (train):")
    all_stats = defaultdict(int)
    for cls_id, count in sorted(train_stats.items()):
        logger.info(f"  {cls_id}: {VITTA_CLASS_NAMES.get(cls_id, '?'):12s} = {count:,}")
        all_stats[cls_id] += count
    logger.info("")
    logger.info("Class distribution (val):")
    for cls_id, count in sorted(val_stats.items()):
        logger.info(f"  {cls_id}: {VITTA_CLASS_NAMES.get(cls_id, '?'):12s} = {count:,}")
        all_stats[cls_id] += count
    logger.info("")
    logger.info("Combined total per class:")
    for cls_id, count in sorted(all_stats.items()):
        logger.info(f"  {cls_id}: {VITTA_CLASS_NAMES.get(cls_id, '?'):12s} = {count:,}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
