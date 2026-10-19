#!/usr/bin/env python3
"""Standalone ONNX inference without the custom YOLOv13 Python source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from onnx_postprocess import decode_predictions, preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, action="append", required=True)
    parser.add_argument("--classes", type=Path, default=Path(__file__).with_name("classes.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def providers(device: str) -> list[str]:
    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def read_image(path: Path) -> np.ndarray | None:
    """Read Unicode paths reliably on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> None:
    args = parse_args()
    names = json.loads(args.classes.read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(args.model), providers=providers(args.device))
    input_name = session.get_inputs()[0].name
    for image_path in args.image:
        image = read_image(image_path)
        if image is None:
            raise FileNotFoundError(image_path)
        tensor, ratio, padding = preprocess(image, args.imgsz)
        output = session.run(None, {input_name: tensor})[0]
        detections = decode_predictions(
            output,
            ratio,
            padding,
            image.shape[:2],
            names,
            confidence_threshold=args.conf,
            iou_threshold=args.iou,
            max_det=args.max_det,
        )
        payload = {
            "image": str(image_path.resolve()),
            "model": str(args.model.resolve()),
            "coordinate_system": "top_left_xyxy",
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "detections": detections,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / f"{image_path.stem}_yolo_raw.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{image_path.name}: {len(detections)} detections -> {output_path}")


if __name__ == "__main__":
    main()
