#!/usr/bin/env python3
"""Run PP-OCRv6 medium on PCB images and save OCR JSON/visualizations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[2]
DEFAULT_VAL_DIR = PROJECT_ROOT / "outputs" / "yolo_components" / "images" / "val"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "OCR_V6_925"
DEFAULT_CACHE_DIR = ROOT / ".paddlex"
DEFAULT_DET_MODEL_DIR = ROOT / "models" / "det"
DEFAULT_REC_MODEL_DIR = ROOT / "models" / "rec"


def jsonable(value: Any) -> Any:
    """Convert numpy values returned by PaddleOCR into JSON-compatible values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def result_field(result: Any, field: str, default: Any) -> Any:
    try:
        return result[field]
    except (KeyError, TypeError, IndexError):
        pass

    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict):
        return payload.get(field, default)
    return default


def create_ocr(
    det_model_dir: Path = DEFAULT_DET_MODEL_DIR,
    rec_model_dir: Path = DEFAULT_REC_MODEL_DIR,
    device: str = "gpu:0",
    box_thresh: float = 0.30,
    rec_score_thresh: float = 0.0,
    limit_side_len: int = 4000,
    textline_orientation: bool = True,
) -> Any:
    """Create one reusable PP-OCRv6 instance for an integrated pipeline."""
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(DEFAULT_CACHE_DIR))

    from paddleocr import PaddleOCR

    ocr_kwargs = {
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": textline_orientation,
        "text_det_limit_side_len": limit_side_len,
        "text_det_limit_type": "max",
        "text_det_box_thresh": box_thresh,
        "text_rec_score_thresh": rec_score_thresh,
        "device": device,
        "enable_mkldnn": False,
        "engine": "paddle",
        "text_detection_model_dir": str(det_model_dir.resolve()),
        "text_recognition_model_dir": str(rec_model_dir.resolve()),
    }
    return PaddleOCR(**ocr_kwargs)


def run_one(ocr: Any, image_path: Path, output_dir: Path) -> None:
    case = image_path.stem
    case_dir = output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)

    results = ocr.predict(str(image_path))
    if not results:
        raise RuntimeError(f"PP-OCRv6 returned no result for {image_path}")

    result = results[0]
    texts = result_field(result, "rec_texts", [])
    scores = result_field(result, "rec_scores", [])
    boxes = result_field(result, "rec_boxes", [])
    polygons = result_field(result, "rec_polys", [])

    items = []
    for index, text in enumerate(texts):
        text = str(text).strip()
        if not text:
            continue
        score = float(scores[index]) if index < len(scores) else None
        bbox = jsonable(boxes[index]) if index < len(boxes) else None
        polygon = jsonable(polygons[index]) if index < len(polygons) else None
        items.append({"text": text, "score": score, "bbox": bbox, "polygon": polygon})

    payload = {
        "case": case,
        "source_image": str(image_path),
        "model": "PP-OCRv6_medium",
        "texts": items,
    }
    (case_dir / f"{case}_ocr.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # PaddleOCR writes an annotated image and its native result JSON here.
    result.save_to_img(str(case_dir))
    result.save_to_json(str(case_dir))
    print(f"{case}: {len(items)} text regions -> {case_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PP-OCRv6 medium on PCB images")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--image", type=Path, help="One image path")
    group.add_argument("--case", help="Validation case ID, e.g. 0003")
    group.add_argument("--all", action="store_true", help="Run all validation images")
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--det-model-dir",
        type=Path,
        default=DEFAULT_DET_MODEL_DIR,
        help="Exported custom detection model directory containing inference.yml.",
    )
    parser.add_argument(
        "--rec-model-dir",
        type=Path,
        default=DEFAULT_REC_MODEL_DIR,
        help="Exported custom recognition model directory containing inference.yml.",
    )
    parser.add_argument("--device", default="gpu:0", help="Paddle device, e.g. gpu:0 or cpu")
    parser.add_argument("--box-thresh", type=float, default=0.30)
    parser.add_argument("--rec-score-thresh", type=float, default=0.0)
    parser.add_argument("--limit-side-len", type=int, default=4000)
    parser.add_argument(
        "--textline-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Classify text-line orientation before recognition; needed for rotated pin labels.",
    )
    args = parser.parse_args()
    if not (args.image or args.case or args.all):
        args.case = "0003"
    return args


def main() -> None:
    args = parse_args()

    ocr = create_ocr(
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        device=args.device,
        box_thresh=args.box_thresh,
        rec_score_thresh=args.rec_score_thresh,
        limit_side_len=args.limit_side_len,
        textline_orientation=args.textline_orientation,
    )

    if args.image:
        images = [args.image]
    elif args.case:
        images = [args.val_dir / f"{args.case}.png"]
    else:
        images = sorted(args.val_dir.glob("*.png"))

    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image(s) not found: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        run_one(ocr, image_path, args.output)


if __name__ == "__main__":
    main()
