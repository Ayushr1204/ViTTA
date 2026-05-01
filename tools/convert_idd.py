"""
ViTTA — IDD Detection Dataset Converter

Converts IDD Detection from Pascal VOC XML → YOLO format, remapping
the IDD class names to ViTTA's 8-class scheme.

IDD Detection classes (from XML annotations):
  car, bus, truck, autorickshaw, motorcycle, bicycle, person, rider,
  vehicle fallback, animal, traffic sign, train

ViTTA classes (YOLO, 0-indexed):
  0:Car  1:Bus  2:Truck  3:Auto  4:2W  5:LCV  6:Bicycle  7:Pedestrian

Mapping:
  car              → 0 (Car)
  bus              → 1 (Bus)
  truck            → 2 (Truck)
  autorickshaw     → 3 (Auto)
  motorcycle       → 4 (2W)
  bicycle          → 6 (Bicycle)
  person, rider    → 7 (Pedestrian)
  vehicle fallback → 0 (Car)  — most common fallback in IDD
  animal, traffic sign, train → SKIP

Usage:
    python convert_idd.py
    python convert_idd.py --help
"""

import argparse
import logging
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert_idd")

# ── Class remapping ───────────────────────────────────────────────────
IDD_TO_VITTA = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "autorickshaw": 3,
    "motorcycle": 4,
    "bicycle": 6,
    "person": 7,
    "rider": 7,
    "vehicle fallback": 0,
    # Skipped:
    "animal": None,
    "traffic sign": None,
    "train": None,
}

VITTA_CLASS_NAMES = {
    0: "Car", 1: "Bus", 2: "Truck", 3: "Auto",
    4: "2W", 5: "LCV", 6: "Bicycle", 7: "Pedestrian",
}


def convert_idd(
    dataset_dir: Path,
    split_file: Path,
    out_img_dir: Path,
    out_lbl_dir: Path,
    prefix: str,
) -> tuple:
    """Convert one split of IDD Detection from VOC XML → YOLO format."""
    ann_base = dataset_dir / "Annotations"
    img_base = dataset_dir / "JPEGImages"

    # Read the split file to get list of image identifiers
    with open(split_file, "r") as f:
        entries = [line.strip() for line in f if line.strip()]

    logger.info(f"  Split file: {split_file.name} ({len(entries)} entries)")

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    skipped_cls = defaultdict(int)
    converted = 0
    no_xml = 0
    no_img = 0

    for entry in entries:
        # Entry format: "frontFar/BLR-2018-03-22_17-39-26_2_frontFar/001542_r"
        xml_path = ann_base / f"{entry}.xml"
        if not xml_path.exists():
            no_xml += 1
            continue

        # Parse XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        # Get image size
        size_el = root.find("size")
        if size_el is None:
            continue
        img_w = int(size_el.find("width").text)
        img_h = int(size_el.find("height").text)

        if img_w <= 0 or img_h <= 0:
            continue

        # Find matching image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = img_base / f"{entry}{ext}"
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            no_img += 1
            continue

        # Convert objects
        yolo_lines = []
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None:
                continue
            cls_name = name_el.text.strip().lower()

            vitta_cls = IDD_TO_VITTA.get(cls_name)
            if vitta_cls is None:
                skipped_cls[cls_name] += 1
                continue

            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue

            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Ensure valid box
            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert to YOLO normalized format
            cx = ((xmin + xmax) / 2.0) / img_w
            cy = ((ymin + ymax) / 2.0) / img_h
            nw = (xmax - xmin) / img_w
            nh = (ymax - ymin) / img_h

            # Clamp
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            if nw < 0.001 or nh < 0.001:
                continue

            yolo_lines.append(f"{vitta_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            stats[vitta_cls] += 1

        # Create unique filename from the path structure
        # e.g., "frontFar/BLR-2018-03-22_17-39-26_2_frontFar/001542_r"
        # → "idd_frontFar_BLR-2018-03-22_17-39-26_2_frontFar_001542_r"
        safe_name = f"{prefix}_{entry.replace('/', '_').replace(chr(92), '_')}"

        # Copy image
        dst_img = out_img_dir / f"{safe_name}{img_path.suffix}"
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)

        # Write label
        with open(out_lbl_dir / f"{safe_name}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

        converted += 1
        if converted % 5000 == 0:
            logger.info(f"    Converted {converted} images...")

    logger.info(f"    Done: {converted} images converted")
    if no_xml:
        logger.info(f"    {no_xml} entries had no XML annotation")
    if no_img:
        logger.info(f"    {no_img} entries had no matching image")
    if skipped_cls:
        logger.info(f"    Skipped classes: {dict(skipped_cls)}")

    return dict(stats), converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert IDD Detection (Pascal VOC XML) → YOLO format with ViTTA class remapping",
    )
    parser.add_argument(
        "--dataset-dir", type=str,
        default="datasets/idd-detection/IDD_Detection",
        help="Path to the IDD_Detection root (containing Annotations/, JPEGImages/).",
    )
    parser.add_argument(
        "--output", "-o", type=str,
        default="datasets/idd_yolo",
        help="Output directory for the YOLO-format dataset.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output)

    logger.info("=" * 60)
    logger.info("IDD Detection → YOLO Converter")
    logger.info("=" * 60)

    # Verify paths
    for p in [dataset_dir / "Annotations", dataset_dir / "JPEGImages"]:
        if not p.exists():
            logger.error(f"Not found: {p}")
            sys.exit(1)

    t_start = time.perf_counter()

    # Convert train split
    train_file = dataset_dir / "train.txt"
    val_file = dataset_dir / "val.txt"

    if not train_file.exists() or not val_file.exists():
        logger.error("train.txt or val.txt not found in dataset directory")
        sys.exit(1)

    logger.info("Converting TRAIN split...")
    train_stats, train_count = convert_idd(
        dataset_dir, train_file,
        out_dir / "images" / "train",
        out_dir / "labels" / "train",
        prefix="idd",
    )

    logger.info("Converting VAL split...")
    val_stats, val_count = convert_idd(
        dataset_dir, val_file,
        out_dir / "images" / "val",
        out_dir / "labels" / "val",
        prefix="idd",
    )

    elapsed = time.perf_counter() - t_start

    # Write data.yaml
    yaml_content = f"""# IDD Detection converted to ViTTA 8-class YOLO format

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
