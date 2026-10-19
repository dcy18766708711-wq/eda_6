"""Preprocessing, decoding, coordinate scaling and class-aware NMS for YOLO ONNX."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def letterbox(image: np.ndarray, image_size: int = 1280) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(image_size / height, image_size / width)
    resized_width = int(round(width * ratio))
    resized_height = int(round(height * ratio))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (image_size - resized_width) / 2.0
    pad_y = (image_size - resized_height) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    right = image_size - resized_width - left
    bottom = image_size - resized_height - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, ratio, (float(left), float(top))


def preprocess(image: np.ndarray, image_size: int = 1280) -> tuple[np.ndarray, float, tuple[float, float]]:
    padded, ratio, padding = letterbox(image, image_size)
    tensor = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[None, ...], ratio, padding


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = np.empty_like(boxes)
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return output


def iou_one(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    left = np.maximum(box[0], others[:, 0])
    top = np.maximum(box[1], others[:, 1])
    right = np.minimum(box[2], others[:, 2])
    bottom = np.minimum(box[3], others[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_box = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_others = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(0.0, others[:, 3] - others[:, 1])
    return intersection / np.maximum(area_box + area_others - intersection, 1e-9)


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float) -> np.ndarray:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        candidates = np.flatnonzero(class_ids == class_id)
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
        while len(candidates):
            current = int(candidates[0])
            kept.append(current)
            if len(candidates) == 1:
                break
            overlaps = iou_one(boxes[current], boxes[candidates[1:]])
            candidates = candidates[1:][overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def decode_predictions(
    output: np.ndarray,
    ratio: float,
    padding: tuple[float, float],
    original_shape: tuple[int, int],
    class_names: dict[str, str],
    confidence_threshold: float = 0.25,
    class_confidence_thresholds: dict[str, float] | None = None,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    # This exported model has shape [1, 4 + num_classes, candidates].
    prediction = np.asarray(output)[0].T
    boxes = xywh_to_xyxy(prediction[:, :4])
    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    thresholds = np.full(scores.shape, float(confidence_threshold), dtype=np.float32)
    if class_confidence_thresholds:
        for class_id, class_name in class_names.items():
            try:
                threshold = float(class_confidence_thresholds.get(str(class_name).lower(), confidence_threshold))
            except (TypeError, ValueError):
                threshold = float(confidence_threshold)
            thresholds[class_ids == int(class_id)] = threshold
    keep = scores >= thresholds
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    if not len(boxes):
        return []
    kept = class_aware_nms(boxes, scores, class_ids, iou_threshold)
    kept = kept[np.argsort(scores[kept])[::-1]][:max_det]

    height, width = original_shape
    pad_x, pad_y = padding
    boxes = boxes[kept]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

    result = []
    for box, score, class_id in zip(boxes, scores[kept], class_ids[kept]):
        result.append(
            {
                "class_name": class_names[str(int(class_id))],
                "confidence": round(float(score), 6),
                "bbox_xyxy_top_left": [round(float(value), 2) for value in box],
            }
        )
    return result


def _box_iou_lists(left: list[float], right: list[float]) -> float:
    """IoU for two xyxy boxes in the same image coordinate frame."""
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(1e-9, left_area + right_area - intersection)


def weighted_box_fusion(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.55,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    """Fuse overlapping detections of the same class from full/tile views.

    This is deliberately class-aware and confidence-weighted.  It is used
    only after each view has run its own NMS, so a tile cannot suppress a
    different class or a distant object.  Every fused record retains the
    candidate IDs and source regions for auditability.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for detection in detections:
        class_name = str(detection.get("class_name", "")).lower()
        box = detection.get("bbox_xyxy_top_left")
        if not class_name or not isinstance(box, list) or len(box) != 4:
            continue
        grouped.setdefault(class_name, []).append(detection)

    fused: list[dict[str, Any]] = []
    for class_name, items in grouped.items():
        remaining = sorted(items, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        while remaining:
            seed = remaining.pop(0)
            cluster = [seed]
            keep: list[dict[str, Any]] = []
            for candidate in remaining:
                if _box_iou_lists(seed["bbox_xyxy_top_left"], candidate["bbox_xyxy_top_left"]) >= iou_threshold:
                    cluster.append(candidate)
                else:
                    keep.append(candidate)
            remaining = keep

            weights = np.asarray(
                [max(1e-6, float(item.get("confidence", 0.0))) for item in cluster],
                dtype=np.float64,
            )
            boxes = np.asarray([item["bbox_xyxy_top_left"] for item in cluster], dtype=np.float64)
            fused_box = (boxes * weights[:, None]).sum(axis=0) / max(1e-9, float(weights.sum()))
            result = dict(seed)
            result["class_name"] = seed.get("class_name", class_name)
            result["confidence"] = round(max(float(item.get("confidence", 0.0)) for item in cluster), 6)
            result["bbox_xyxy_top_left"] = [round(float(value), 2) for value in fused_box]
            result["fusion_count"] = len(cluster)
            result["fusion_member_ids"] = [item.get("candidate_id") for item in cluster if item.get("candidate_id")]
            result["fusion_sources"] = sorted({str(item.get("source", "unknown")) for item in cluster})
            fused.append(result)

    fused.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    return fused[:max_det]
