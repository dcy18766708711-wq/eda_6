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
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    # This exported model has shape [1, 4 + num_classes, candidates].
    prediction = np.asarray(output)[0].T
    boxes = xywh_to_xyxy(prediction[:, :4])
    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
    scores = class_scores[np.arange(len(class_scores)), class_ids]
    keep = scores >= confidence_threshold
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
