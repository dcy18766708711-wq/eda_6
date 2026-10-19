#!/usr/bin/env python3
"""Run the local YOLO ONNX detector and PP-OCRv6 in one pipeline.

The intermediate result uses top-left image coordinates, which are native to
OpenCV, ONNX post-processing and PaddleOCR. A later contest-output adapter
must convert all points and boxes to the required bottom-left coordinates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT / "onnx_yolo"
OCR_DIR = ROOT / "ppocrv6"
sys.path.insert(0, str(YOLO_DIR))
sys.path.insert(0, str(OCR_DIR))

from onnx_postprocess import decode_predictions, preprocess, weighted_box_fusion  # noqa: E402
from onnx_predict import providers, read_image  # noqa: E402
from run_ppocrv6_medium import create_ocr, jsonable, result_field  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, action="append", help="One or more image paths")
    inputs.add_argument("--input-dir", type=Path, help="Directory containing image files")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "yolo_ocr")
    parser.add_argument("--yolo-model", type=Path, default=YOLO_DIR / "models" / "best.onnx")
    parser.add_argument("--classes", type=Path, default=YOLO_DIR / "classes.json")
    parser.add_argument("--yolo-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ocr-device", default="cpu", help="Paddle device, e.g. cpu or gpu:0")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--enable-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run supplementary high-resolution tile inference and fuse it with full-image detections.",
    )
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--tile-fusion-iou", type=float, default=0.55)
    parser.add_argument("--tile-suppression-iou", type=float, default=0.30)
    parser.add_argument("--tile-supplement-conf", type=float, default=0.30)
    parser.add_argument("--tile-single-view-conf", type=float, default=0.45)
    parser.add_argument("--tile-consistency-iou", type=float, default=0.50)
    parser.add_argument("--tile-boundary-margin-ratio", type=float, default=0.01)
    parser.add_argument(
        "--adaptive-tile-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scale the tile side from the image size so smaller cases also receive local inference.",
    )
    parser.add_argument("--tile-min-size", type=int, default=384)
    parser.add_argument(
        "--small-object-classes",
        default="r,c,gnd,v",
        help="Comma-separated class names that use the lower small-object threshold.",
    )
    parser.add_argument("--small-object-conf", type=float, default=0.12)
    parser.add_argument("--det-model-dir", type=Path, default=OCR_DIR / "models" / "det")
    parser.add_argument("--rec-model-dir", type=Path, default=OCR_DIR / "models" / "rec")
    parser.add_argument("--ocr-box-thresh", type=float, default=0.30)
    parser.add_argument("--ocr-rec-score-thresh", type=float, default=0.0)
    parser.add_argument("--ocr-limit-side-len", type=int, default=4000)
    parser.add_argument(
        "--textline-orientation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PP-OCRv6 text-line orientation classification.",
    )
    return parser.parse_args()


def get_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.image:
        paths = [path.resolve() for path in args.image]
    else:
        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        paths = sorted(
            path.resolve()
            for path in args.input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in allowed
        )
    if not paths:
        raise FileNotFoundError("No input images found")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Image(s) not found: " + ", ".join(missing))
    return paths


def as_xyxy_top_left(value: Any) -> list[float] | None:
    """Normalize PaddleOCR boxes/polygons to [x1, y1, x2, y2]."""
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size < 4:
        return None
    if array.ndim >= 2 and array.shape[-1] == 2:
        points = array.reshape(-1, 2)
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
    else:
        flat = array.reshape(-1)
        x1, y1, x2, y2 = flat[:4]
    return [round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)]


def read_ocr_items(ocr_result: Any) -> list[dict[str, Any]]:
    texts = result_field(ocr_result, "rec_texts", [])
    scores = result_field(ocr_result, "rec_scores", [])
    boxes = result_field(ocr_result, "rec_boxes", [])
    polygons = result_field(ocr_result, "rec_polys", [])
    items: list[dict[str, Any]] = []
    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        box_value = boxes[index] if index < len(boxes) else None
        polygon_value = polygons[index] if index < len(polygons) else None
        box = as_xyxy_top_left(jsonable(box_value))
        polygon = jsonable(polygon_value) if polygon_value is not None else None
        score = float(scores[index]) if index < len(scores) else None
        items.append(
            {
                "text": text,
                "score": round(score, 6) if score is not None else None,
                "bbox_xyxy_top_left": box,
                "polygon_top_left": polygon,
                "source": "ppocrv6_full_image",
            }
        )
    return items


def rectangle_distance(a: list[float], b: list[float]) -> float:
    """Distance between two rectangles; zero when they overlap."""
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return float((dx * dx + dy * dy) ** 0.5)


def intersection_over_area(inner: list[float], outer: list[float]) -> float:
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    inner_area = max(1.0, inner[2] - inner[0]) * max(1.0, inner[3] - inner[1])
    return float(area / inner_area)


def associate_texts(
    detections: list[dict[str, Any]], texts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Create ranked text candidates for each component detection.

    This is a candidate association, not final semantic assignment. Text can
    be above or below a symbol, so a soft spatial rule is safer than requiring
    an OCR box to lie inside the component bbox.
    """
    links: list[dict[str, Any]] = []
    for component_index, detection in enumerate(detections):
        component_box = detection["bbox_xyxy_top_left"]
        width = max(1.0, component_box[2] - component_box[0])
        height = max(1.0, component_box[3] - component_box[1])
        max_distance = max(80.0, min(300.0, 1.5 * max(width, height)))
        expanded = [
            component_box[0] - 0.35 * width,
            component_box[1] - 0.75 * height,
            component_box[2] + 0.35 * width,
            component_box[3] + 0.75 * height,
        ]
        candidates = []
        for text_index, text_item in enumerate(texts):
            text_box = text_item.get("bbox_xyxy_top_left")
            if not text_box:
                continue
            distance = rectangle_distance(text_box, component_box)
            overlap = intersection_over_area(text_box, expanded)
            if distance > max_distance and overlap <= 0.0:
                continue
            score = min(1.0, 0.65 * overlap + 0.35 * max(0.0, 1.0 - distance / max_distance))
            candidates.append(
                {
                    "text_index": text_index,
                    "association_score": round(float(score), 6),
                    "distance_px": round(distance, 2),
                }
            )
        candidates.sort(key=lambda item: item["association_score"], reverse=True)
        links.append({"component_index": component_index, "text_candidates": candidates[:8]})
    return links


def create_yolo_session(model_path: Path, device: str) -> tuple[ort.InferenceSession, str]:
    session = ort.InferenceSession(str(model_path.resolve()), providers=providers(device))
    return session, session.get_inputs()[0].name


def effective_tile_size(image_shape: tuple[int, int], args: argparse.Namespace) -> int:
    """Choose a scale-independent tile side without using case-specific values."""
    height, width = image_shape[:2]
    requested = max(256, int(getattr(args, "tile_size", 960)))
    if not bool(getattr(args, "adaptive_tile_size", True)):
        return requested
    minimum = max(256, int(getattr(args, "tile_min_size", 384)))
    longest_side = max(int(height), int(width))
    scaled = int(round(0.8 * longest_side))
    return max(minimum, min(requested, scaled))


def run_yolo(
    session: ort.InferenceSession,
    input_name: str,
    image: np.ndarray,
    names: dict[str, str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    small_classes = {
        item.strip().lower()
        for item in str(args.small_object_classes).split(",")
        if item.strip()
    }
    class_thresholds = {name: float(args.small_object_conf) for name in small_classes}

    def infer_view(view: np.ndarray, offset_x: int, offset_y: int, source: str, region: list[int]) -> list[dict[str, Any]]:
        tensor, ratio, padding = preprocess(view, args.imgsz)
        raw_output = session.run(None, {input_name: tensor})[0]
        decoded = decode_predictions(
            raw_output,
            ratio,
            padding,
            view.shape[:2],
            names,
            confidence_threshold=args.conf,
            class_confidence_thresholds=class_thresholds,
            iou_threshold=args.iou,
            max_det=args.max_det,
        )
        for item in decoded:
            box = item["bbox_xyxy_top_left"]
            item["bbox_xyxy_top_left"] = [
                round(float(box[0] + offset_x), 2),
                round(float(box[1] + offset_y), 2),
                round(float(box[2] + offset_x), 2),
                round(float(box[3] + offset_y), 2),
            ]
            item["source"] = source
            item["inference_region_xyxy"] = region
        return decoded

    height, width = image.shape[:2]
    full_candidates: list[dict[str, Any]] = []
    tile_candidates: list[dict[str, Any]] = []
    full_region = [0, 0, int(width), int(height)]
    full_candidates.extend(infer_view(image, 0, 0, "full_image", full_region))

    tile_regions: list[list[int]] = []
    tile_size = effective_tile_size(image.shape[:2], args)
    overlap = min(0.8, max(0.05, float(getattr(args, "tile_overlap", 0.25))))
    if args.enable_tiles and (width > tile_size or height > tile_size):
        step = max(1, int(round(tile_size * (1.0 - overlap))))

        def starts(length: int) -> list[int]:
            if length <= tile_size:
                return [0]
            values = list(range(0, length - tile_size + 1, step))
            last = length - tile_size
            if values[-1] != last:
                values.append(last)
            return values

        for y0 in starts(height):
            for x0 in starts(width):
                x1 = min(width, x0 + tile_size)
                y1 = min(height, y0 + tile_size)
                tile_regions.append([int(x0), int(y0), int(x1), int(y1)])
        for tile_id, (x0, y0, x1, y1) in enumerate(tile_regions):
            tile_candidates.extend(
                infer_view(
                    image[y0:y1, x0:x1],
                    x0,
                    y0,
                    f"tile_{tile_id:03d}",
                    [x0, y0, x1, y1],
                )
            )

    candidates = full_candidates + tile_candidates
    for index, item in enumerate(candidates, start=1):
        item["candidate_id"] = f"candidate_{index:05d}"
        item["lifecycle_status"] = "local_nms_kept"

    def box_iou(left: list[float], right: list[float]) -> float:
        ix1 = max(float(left[0]), float(right[0]))
        iy1 = max(float(left[1]), float(right[1]))
        ix2 = min(float(left[2]), float(right[2]))
        iy2 = min(float(left[3]), float(right[3]))
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
        right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
        return intersection / max(1e-9, left_area + right_area - intersection)

    suppression_iou = float(getattr(args, "tile_suppression_iou", 0.30))
    supplement_conf = float(getattr(args, "tile_supplement_conf", 0.30))
    single_view_conf = float(getattr(args, "tile_single_view_conf", 0.45))
    consistency_iou = float(getattr(args, "tile_consistency_iou", 0.50))
    boundary_margin_ratio = float(getattr(args, "tile_boundary_margin_ratio", 0.01))
    accepted_tiles: list[dict[str, Any]] = []
    for tile in tile_candidates:
        confidence = float(tile.get("confidence", 0.0))
        if confidence < supplement_conf:
            tile["lifecycle_status"] = "rejected_by_tile_confidence"
            continue
        blocked = any(
            str(tile.get("class_name", "")).lower() == str(full.get("class_name", "")).lower()
            and box_iou(tile["bbox_xyxy_top_left"], full["bbox_xyxy_top_left"]) >= suppression_iou
            for full in full_candidates
        )
        if blocked:
            tile["lifecycle_status"] = "rejected_by_full_priority"
        else:
            region = tile.get("inference_region_xyxy")
            box = tile.get("bbox_xyxy_top_left")
            if isinstance(region, list) and len(region) == 4 and isinstance(box, list) and len(box) == 4:
                tile_width = max(1.0, float(region[2]) - float(region[0]))
                tile_height = max(1.0, float(region[3]) - float(region[1]))
                boundary_margin = max(
                    4.0,
                    boundary_margin_ratio * max(tile_width, tile_height),
                )
                tile["tile_boundary_margin_px"] = round(boundary_margin, 3)
                tile["tile_boundary_risk"] = bool(
                    min(
                        float(box[0]) - float(region[0]),
                        float(region[2]) - float(box[2]),
                        float(box[1]) - float(region[1]),
                        float(region[3]) - float(box[3]),
                    ) < boundary_margin
                )
            else:
                tile["tile_boundary_risk"] = True
            accepted_tiles.append(tile)

    tile_supplements = weighted_box_fusion(
        accepted_tiles,
        iou_threshold=consistency_iou,
        max_det=max(0, int(args.max_det) - len(full_candidates)),
    )
    accepted_supplements: list[dict[str, Any]] = []
    supplement_ids: set[str] = set()
    for supplement in tile_supplements:
        member_ids = [str(value) for value in supplement.get("fusion_member_ids", []) if value]
        member_map = {
            str(item.get("candidate_id")): item
            for item in accepted_tiles
            if item.get("candidate_id")
        }
        members = [member_map[item_id] for item_id in member_ids if item_id in member_map]
        source_count = len({str(item.get("source")) for item in members})
        confidence = float(supplement.get("confidence", 0.0))
        has_multi_tile_support = source_count >= 2
        all_interior = bool(members) and all(not item.get("tile_boundary_risk", True) for item in members)
        single_view_exception = source_count == 1 and all_interior and confidence >= single_view_conf
        supplement["tile_view_count"] = source_count
        supplement["tile_consistency_iou"] = consistency_iou
        supplement["tile_multi_view_support"] = has_multi_tile_support
        if not (has_multi_tile_support or single_view_exception):
            reason = "rejected_by_tile_boundary" if not all_interior else "rejected_by_tile_consistency"
            for member in members:
                member["lifecycle_status"] = reason
            continue
        accepted_supplements.append(supplement)
        supplement_ids.update(member_ids)
    tile_supplements = accepted_supplements
    final = full_candidates[: int(args.max_det)] + tile_supplements
    final_ids: set[str] = set()
    for index, item in enumerate(final, start=1):
        item["detection_id"] = f"yolo_{index:04d}"
        item["lifecycle_status"] = "final_kept"
        final_ids.update(str(value) for value in item.get("fusion_member_ids", []))
    for item in candidates:
        candidate_id = item.get("candidate_id")
        if item.get("source") == "full_image":
            if candidate_id in {full.get("candidate_id") for full in final}:
                item["lifecycle_status"] = "full_image_final"
            else:
                item["lifecycle_status"] = "removed_by_max_det"
        elif item.get("lifecycle_status") == "rejected_by_full_priority":
            continue
        elif item.get("lifecycle_status", "").startswith("rejected_by_tile_"):
            continue
        elif candidate_id in supplement_ids or candidate_id in final_ids:
            item["lifecycle_status"] = "fused_to_tile_supplement"
        else:
            item["lifecycle_status"] = "removed_by_tile_fusion_or_max_det"
    return final, candidates


def run_one(
    image_path: Path,
    output_dir: Path,
    yolo_session: ort.InferenceSession,
    yolo_input_name: str,
    ocr: Any,
    names: dict[str, str],
    args: argparse.Namespace,
) -> Path:
    image = read_image(image_path)
    if image is None:
        raise RuntimeError(f"OpenCV cannot read image: {image_path}")

    detections, detection_audit = run_yolo(yolo_session, yolo_input_name, image, names, args)
    # Passing the decoded array avoids Unicode-path issues inside OpenCV/Paddle.
    ocr_results = ocr.predict(image)
    if not ocr_results:
        raise RuntimeError(f"PP-OCRv6 returned no result for {image_path}")
    texts = read_ocr_items(ocr_results[0])

    payload = {
        "pipeline": "yolo_onnx_plus_ppocrv6",
        "image": str(image_path),
        "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
        "coordinate_system": "top_left_xyxy",
        "yolo": {
            "model": str(args.yolo_model.resolve()),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "tiles_enabled": bool(args.enable_tiles),
            "tile_size": int(args.tile_size),
            "tile_overlap": float(args.tile_overlap),
            "tile_fusion_iou": float(args.tile_fusion_iou),
            "tile_suppression_iou": float(getattr(args, "tile_suppression_iou", 0.30)),
            "tile_supplement_conf": float(getattr(args, "tile_supplement_conf", 0.30)),
            "tile_single_view_conf": float(getattr(args, "tile_single_view_conf", 0.45)),
            "tile_consistency_iou": float(getattr(args, "tile_consistency_iou", 0.50)),
            "tile_boundary_margin_ratio": float(getattr(args, "tile_boundary_margin_ratio", 0.01)),
            "tile_priority_policy": "full_image_authoritative_tile_nonoverlap_supplement",
            "adaptive_tile_size": bool(getattr(args, "adaptive_tile_size", True)),
            "tile_min_size": int(getattr(args, "tile_min_size", 384)),
            "effective_tile_size": effective_tile_size(image.shape[:2], args),
            "small_object_classes": sorted(
                item.strip().lower()
                for item in str(args.small_object_classes).split(",")
                if item.strip()
            ),
            "small_object_conf": float(args.small_object_conf),
            "detections": detections,
            "detection_lifecycle": detection_audit,
        },
        "ocr": {"model": "PP-OCRv6_medium", "texts": texts},
        "component_text_links": associate_texts(detections, texts),
        "next_stage_note": (
            "Intermediate result only: it does not yet contain final components, "
            "pins or nets and has not been converted to bottom-left coordinates."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_yolo_ocr.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{image_path.name}: {len(detections)} components, {len(texts)} texts -> {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    paths = get_image_paths(args)
    names = json.loads(args.classes.read_text(encoding="utf-8"))
    yolo_session, yolo_input_name = create_yolo_session(args.yolo_model, args.yolo_device)
    ocr = create_ocr(
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        device=args.ocr_device,
        box_thresh=args.ocr_box_thresh,
        rec_score_thresh=args.ocr_rec_score_thresh,
        limit_side_len=args.ocr_limit_side_len,
        textline_orientation=args.textline_orientation,
    )
    for path in paths:
        run_one(path, args.output_dir, yolo_session, yolo_input_name, ocr, names, args)


if __name__ == "__main__":
    main()
