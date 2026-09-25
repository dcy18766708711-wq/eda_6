#!/usr/bin/env python3
"""Evaluate YOLOv13 component predictions against the original contest JSON.

This is an additional sanity check for the 40 validation cases. Ultralytics'
official mAP is still the main detector metric; this script gives an explicit
JSON-to-prediction comparison at a chosen confidence and IoU threshold.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/dcy2002/EDA_6/outputs/yolo_components")
SOURCE_ROOT = Path("/home/dcy2002/EDA_6/赛题六公开数据集/200_train_cases")
BEST_MODEL = PROJECT_ROOT / "outputs/yolov13_components/yolov13n_eda6_components/weights/best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs/yolov13_components/final_validation/json_evaluation"


def iou(box_a: list[float], box_b: list[float]) -> float:
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def find_source_json(case_id: str) -> Path:
    candidates = sorted((SOURCE_ROOT / case_id).glob("*_target_named.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one JSON annotation for case {case_id}, found {candidates}")
    return candidates[0]


def load_classes() -> tuple[list[str], dict[str, int]]:
    payload = json.loads((DATA_ROOT / "classes.json").read_text(encoding="utf-8"))
    classes = [payload["classes"][str(index)] for index in range(len(payload["classes"]))]
    return classes, {name: index for index, name in enumerate(classes)}


def load_ground_truth(case_id: str, class_to_id: dict[str, int], image_height: int) -> list[dict]:
    annotation_path = find_source_json(case_id)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    ground_truth = []
    for key, component in (payload.get("components") or {}).items():
        component_type = str(component.get("type") or "other")
        if component_type not in class_to_id:
            raise ValueError(f"Unknown component type {component_type!r} in {annotation_path}")
        x1, y1, x2, y2 = [float(value) for value in component["bbox"]]
        # Contest JSON uses bottom-left origin; model predictions use image top-left origin.
        box = [x1, image_height - y2, x2, image_height - y1]
        ground_truth.append({"key": key, "class_id": class_to_id[component_type], "bbox": box})
    return ground_truth


def match_predictions(records: list[dict], iou_threshold: float) -> dict:
    true_positive = false_positive = false_negative = 0
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for record in records:
        ground_truth = record["ground_truth"]
        predictions = sorted(record["predictions"], key=lambda item: item["confidence"], reverse=True)
        used = set()
        for prediction in predictions:
            candidates = [
                (iou(prediction["bbox"], target["bbox"]), index, target)
                for index, target in enumerate(ground_truth)
                if index not in used and prediction["class_id"] == target["class_id"]
            ]
            best = max(candidates, default=(0.0, -1, None), key=lambda item: item[0])
            if best[0] >= iou_threshold:
                used.add(best[1])
                true_positive += 1
                per_class[prediction["class_id"]]["tp"] += 1
            else:
                false_positive += 1
                per_class[prediction["class_id"]]["fp"] += 1
        false_negative += len(ground_truth) - len(used)
        for index, target in enumerate(ground_truth):
            if index not in used:
                per_class[target["class_id"]]["fn"] += 1

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": {str(class_id): values for class_id, values in sorted(per_class.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=BEST_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    classes, class_to_id = load_classes()
    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    image_paths = sorted((DATA_ROOT / "images/val").glob("*.png"))
    for image_path in image_paths:
        case_id = image_path.stem
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        ground_truth = load_ground_truth(case_id, class_to_id, image_height)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            max_det=3000,
            device=args.device,
            workers=args.workers,
            verbose=False,
        )[0]
        predictions = []
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            predictions.append(
                {"class_id": int(class_id), "confidence": float(confidence), "bbox": [float(v) for v in box]}
            )
        records.append({"case": case_id, "ground_truth": ground_truth, "predictions": predictions})

    summary = {
        "model": str(args.model),
        "validation_images": len(records),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "classes": {str(index): name for index, name in enumerate(classes)},
        "metrics": match_predictions(records, args.iou),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "predictions_vs_json.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = summary["metrics"]
    print(f"JSON evaluation directory: {args.output.resolve()}")
    print(f"images: {summary['validation_images']}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
    print(f"precision={metrics['precision']:.6f}")
    print(f"recall={metrics['recall']:.6f}")
    print(f"f1={metrics['f1']:.6f}")


if __name__ == "__main__":
    main()
