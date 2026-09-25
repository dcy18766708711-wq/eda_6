#!/usr/bin/env python3
"""Train the server's YOLOv13 model on EDA_6 component annotations.

The converted dataset was created on Windows, so its original data.yaml
contains a Windows path. This script writes a Linux data.yaml under
PCB_analysis before training and does not modify EDA_6/outputs.

Examples:
    /opt/conda/envs/yolov13/bin/python train_yolov13_eda6.py --check
    /opt/conda/envs/yolov13/bin/python train_yolov13_eda6.py
    /opt/conda/envs/yolov13/bin/python train_yolov13_eda6.py --epochs 300 --imgsz 1024 --batch 8
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/home/dcy2002/EDA_6/outputs/yolo_components")
DEFAULT_MODEL = Path("/home/dcy2002/yolov13/yolov13n.pt")
DEFAULT_PROJECT = PROJECT_ROOT / "outputs" / "yolov13_components"
LOCAL_DATA_YAML = PROJECT_ROOT / "configs" / "eda6_yolov13_data.yaml"
LOCAL_YOLO_CONFIG = PROJECT_ROOT / ".yolo_config"
LOCAL_FONT = LOCAL_YOLO_CONFIG / "Arial.ttf"

# This name is only used in plots and logs. The class id and all label files
# stay unchanged, so this does not alter the detector's training targets.
PLOT_NAME_OVERRIDES = {"m3螺丝": "m3_screw"}


def parse_batch(value: str) -> int | float | str:
    if value == "auto":
        return value
    return float(value) if "." in value else int(value)


def load_class_names(data_root: Path) -> list[str]:
    classes_path = data_root / "classes.json"
    if not classes_path.exists():
        raise FileNotFoundError(f"Missing class metadata: {classes_path}")
    payload = json.loads(classes_path.read_text(encoding="utf-8"))
    classes = payload.get("classes", {})
    if not classes:
        raise ValueError(f"No classes found in {classes_path}")
    return [classes[str(index)] for index in range(len(classes))]


def validate_dataset(data_root: Path, class_names: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    class_count = len(class_names)
    for split in ("train", "val"):
        image_dir = data_root / "images" / split
        label_dir = data_root / "labels" / split
        images = sorted(image_dir.glob("*"))
        images = [path for path in images if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        labels = sorted(label_dir.glob("*.txt"))
        if not images:
            raise FileNotFoundError(f"No images found: {image_dir}")
        if len(images) != len(labels):
            raise ValueError(f"{split}: {len(images)} images but {len(labels)} labels")

        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        if image_stems != label_stems:
            raise ValueError(f"{split}: image and label filenames do not match")

        box_count = 0
        for label_path in labels:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO row in {label_path}:{line_number}: {line!r}")
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
                if not 0 <= class_id < class_count:
                    raise ValueError(f"Class id {class_id} out of range in {label_path}:{line_number}")
                if any(value < 0.0 or value > 1.0 for value in values):
                    raise ValueError(f"Normalized box out of range in {label_path}:{line_number}")
                if values[2] <= 0.0 or values[3] <= 0.0:
                    raise ValueError(f"Non-positive box size in {label_path}:{line_number}")
                box_count += 1
        counts[f"{split}_images"] = len(images)
        counts[f"{split}_labels"] = len(labels)
        counts[f"{split}_boxes"] = box_count
    return counts


def write_linux_data_yaml(data_root: Path, class_names: list[str]) -> Path:
    LOCAL_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"path: {data_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for class_id, class_name in enumerate(class_names):
        class_name = PLOT_NAME_OVERRIDES.get(class_name, class_name)
        escaped_name = class_name.replace('"', '\\"')
        lines.append(f'  {class_id}: "{escaped_name}"')
    LOCAL_DATA_YAML.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return LOCAL_DATA_YAML


def prepare_local_font() -> None:
    """Prevent Ultralytics from downloading a font during dataset checks."""
    LOCAL_YOLO_CONFIG.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(LOCAL_YOLO_CONFIG)
    if LOCAL_FONT.exists():
        return
    system_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not system_font.exists():
        raise FileNotFoundError(f"Local fallback font not found: {system_font}")
    shutil.copy2(system_font, LOCAL_FONT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=parse_batch, default=8)
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability.")
    parser.add_argument("--scale", type=float, default=0.5, help="Random scale augmentation range.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", default="yolov13n_eda6_components")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--check", action="store_true", help="Validate data and write YAML without training.")
    parser.add_argument("--resume", type=Path, help="Resume from a YOLOv13 last.pt checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_root}")
    class_names = load_class_names(data_root)
    counts = validate_dataset(data_root, class_names)
    data_yaml = write_linux_data_yaml(data_root, class_names)

    print(f"data root: {data_root}")
    print(f"data yaml: {data_yaml}")
    print(f"classes: {len(class_names)}")
    print(f"dataset counts: {counts}")
    if args.check:
        print("dataset check passed; training was not started")
        return

    if not args.model.exists() and not args.resume:
        raise FileNotFoundError(f"YOLOv13 model not found: {args.model}")
    if args.resume and not args.resume.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    prepare_local_font()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is unavailable. Run this script with:\n"
            "  /opt/conda/envs/yolov13/bin/python train_yolov13_eda6.py"
        ) from exc

    model = YOLO(str(args.resume or args.model))
    train_args = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        deterministic=True,
        optimizer="SGD",
        amp=True,
        close_mosaic=10,
        mosaic=args.mosaic,
        scale=args.scale,
        cache=args.cache,
        plots=True,
        save_period=10,
        exist_ok=True,
    )
    if args.resume:
        train_args["resume"] = True
    model.train(**train_args)


if __name__ == "__main__":
    main()
