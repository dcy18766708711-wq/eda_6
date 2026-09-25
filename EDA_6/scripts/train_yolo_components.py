#!/usr/bin/env python3
"""
Train a YOLO detector for contest-6 component bbox/type detection.

Before running this script, create the YOLO dataset:
    python scripts/convert_components_to_yolo.py --overwrite

Then train:
    python scripts/train_yolo_components.py
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_batch(value: str) -> int | float | str:
    if value == "auto":
        return value
    if "." in value:
        return float(value)
    return int(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Ultralytics YOLO model on component bbox/type labels."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("outputs") / "yolo_components" / "data.yaml",
        help="YOLO data.yaml path.",
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Base model, e.g. yolo11s.pt, yolo11m.pt, yolov8s.pt.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=parse_batch, default=8, help="Batch size or 'auto'.")
    parser.add_argument("--device", default="0", help="GPU id such as 0, or 'cpu'.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--project", default=str(Path("outputs") / "yolo_runs"))
    parser.add_argument("--name", default="components_yolo11s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume previous run.")
    parser.add_argument(
        "--cache",
        default=False,
        action="store_true",
        help="Cache images in RAM/disk as supported by Ultralytics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {args.data}\n"
            "Run: python scripts/convert_components_to_yolo.py --overwrite"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is not installed. Install it on your training server with:\n"
            "  pip install ultralytics\n"
        ) from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        cache=args.cache,
        resume=args.resume,
        exist_ok=True,
    )
    print(results)


if __name__ == "__main__":
    main()
