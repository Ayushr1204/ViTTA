"""
Centralised vehicle class-name mapping for IDD-trained YOLO model.

All modules (CSV writer, visualiser, Excel exporter) import from here
so the mapping is defined exactly once.
"""

# Class ID → human-readable name.
# Must match the order in the IDD YOLO data.yaml used for training.
CLASS_NAMES: dict[int, str] = {
    0: "Car",
    1: "Bus",
    2: "Truck",
    3: "Auto",
    4: "2W",
    5: "LCV",
    6: "Bicycle",
    7: "Pedestrian",
}


def class_name(class_id: int) -> str:
    """Return the human-readable name for a class ID, with a fallback."""
    return CLASS_NAMES.get(class_id, f"cls{class_id}")
