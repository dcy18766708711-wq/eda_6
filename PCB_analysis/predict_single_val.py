#!/usr/bin/env python3
"""Run YOLOv13 on one validation image and save readable visualizations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/dcy2002/EDA_6/outputs/yolo_components")
SOURCE_ROOT = Path("/home/dcy2002/EDA_6/赛题六公开数据集/200_train_cases")
DEFAULT_MODEL = PROJECT_ROOT / "outputs/yolov13_components/yolov13n_eda6_components/weights/best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/yolov13_components/single_validation"

# Avoid any font download if Ultralytics is imported in this process.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".yolo_config"))


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    return ImageFont.truetype(path, size) if path.exists() else ImageFont.load_default()


def find_image(case_id: str) -> Path:
    matches = sorted((DATA_ROOT / "images/val").glob(f"{case_id}.*"))
    matches = [path for path in matches if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one validation image for case {case_id}, found: {matches}")
    return matches[0]


def find_json(case_id: str) -> Path:
    matches = sorted((SOURCE_ROOT / case_id).glob("*_target_named.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one source JSON for case {case_id}, found: {matches}")
    return matches[0]


def load_ground_truth(case_id: str, image_height: int) -> list[dict]:
    payload = json.loads(find_json(case_id).read_text(encoding="utf-8"))
    targets = []
    for key, component in (payload.get("components") or {}).items():
        x1, y1, x2, y2 = [float(value) for value in component["bbox"]]
        targets.append(
            {
                "key": key,
                "type": str(component.get("type") or "other"),
                "Name": component.get("Name"),
                "value": component.get("value"),
                "bbox": [x1, image_height - y2, x2, image_height - y1],
            }
        )
    return targets


def draw_boxes(image: Image.Image, boxes: list[dict], color: tuple[int, int, int], title: str) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    label_font = font(max(12, round(min(output.size) / 90)))
    line_width = max(2, round(min(output.size) / 500))
    for index, item in enumerate(boxes, start=1):
        x1, y1, x2, y2 = [round(value) for value in item["bbox"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = item.get("label", f"{index}")
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_box[2] - text_box[0] + 6
        text_height = text_box[3] - text_box[1] + 4
        label_x = max(0, min(x1, output.width - text_width))
        label_y = max(0, y1 - text_height)
        draw.rectangle((label_x, label_y, label_x + text_width, label_y + text_height), fill=color)
        draw.text((label_x + 3, label_y + 1), label, fill="white", font=label_font)

    header_height = max(34, round(min(output.size) / 22))
    canvas = Image.new("RGB", (output.width, output.height + header_height), "white")
    canvas.paste(output, (0, header_height))
    ImageDraw.Draw(canvas).text((10, 8), f"{title} | boxes: {len(boxes)}", fill="black", font=font(20))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="0003", help="Validation case id, for example 0003.")
    parser.add_argument("--all", action="store_true", help="Process all validation images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [path.stem for path in sorted((DATA_ROOT / "images/val").glob("*.png"))] if args.all else [args.case]
    for case_id in cases:
        image_path = find_image(case_id)
        image = Image.open(image_path).convert("RGB")
        ground_truth = load_ground_truth(case_id, image.height)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            max_det=3000,
            device=args.device,
            verbose=False,
        )[0]

        predictions = []
        for box, confidence, class_id in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
        ):
            class_id = int(class_id)
            class_name = result.names.get(class_id, str(class_id))
            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidence),
                    "bbox": [float(value) for value in box],
                }
            )

        gt_for_drawing = [
            {**target, "label": f"GT {target['key']} {target['type']}"} for target in ground_truth
        ]
        pred_for_drawing = [
            {**prediction, "label": f"{prediction['class_name']} {prediction['confidence']:.2f}"}
            for prediction in predictions
        ]
        gt_image = draw_boxes(image, gt_for_drawing, (220, 40, 40), f"{case_id} ground truth JSON")
        pred_image = draw_boxes(image, pred_for_drawing, (30, 100, 220), f"{case_id} YOLOv13 prediction")
        comparison = Image.new(
            "RGB", (gt_image.width + pred_image.width, max(gt_image.height, pred_image.height)), "white"
        )
        comparison.paste(gt_image, (0, 0))
        comparison.paste(pred_image, (gt_image.width, 0))

        gt_image.save(args.output / f"{case_id}_ground_truth.jpg", quality=95)
        pred_image.save(args.output / f"{case_id}_prediction.jpg", quality=95)
        comparison.save(args.output / f"{case_id}_comparison.jpg", quality=95)
        (args.output / f"{case_id}_prediction.json").write_text(
            json.dumps(
                {
                    "case": case_id,
                    "source_image": str(image_path),
                    "source_json": str(find_json(case_id)),
                    "confidence_threshold": args.conf,
                    "ground_truth": ground_truth,
                    "predictions": predictions,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{case_id}: ground truth={len(ground_truth)}, predictions={len(predictions)}")

    print(f"processed images: {len(cases)}")
    print(f"output directory: {args.output.resolve()}")


if __name__ == "__main__":
    main()
