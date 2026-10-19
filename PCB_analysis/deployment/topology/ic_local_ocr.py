#!/usr/bin/env python3
"""Run a second, component-local PP-OCR pass for IC model text.

Full-page OCR is retained as the primary source.  This module only supplies
additional evidence from a padded, high-resolution crop around each IC box;
all returned boxes are mapped back into full-image top-left coordinates.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _result_field(result: Any, field: str, default: Any) -> Any:
    try:
        return result[field]
    except (KeyError, TypeError, IndexError):
        pass
    payload = getattr(result, "json", None)
    payload = payload() if callable(payload) else payload
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)
    return payload.get(field, default) if isinstance(payload, dict) else default


def create_local_ocr(*, device: str = "cpu") -> Any:
    """Create the project's exported PP-OCRv6 instance once per image."""
    from ppocrv6.run_ppocrv6_medium import create_ocr

    return create_ocr(
        device=device,
        box_thresh=0.15,
        rec_score_thresh=0.0,
        limit_side_len=2400,
        textline_orientation=True,
    )


def _bbox(values: Any) -> list[float] | None:
    if not hasattr(values, "__len__") or len(values) != 4:
        return None
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError):
        return None


def recognize_ic_crop(
    ocr: Any,
    image: np.ndarray,
    bbox_top_left: list[float],
    *,
    margin_ratio: float = 0.22,
    min_margin: int = 60,
    upscale: float = 2.0,
    max_side: int = 1800,
    rotation_degrees: int = 0,
) -> list[dict[str, Any]]:
    """Return local OCR records in original full-image coordinates.

    The crop is enlarged only before OCR.  It does not alter component or pin
    geometry, and recognizer coordinates are converted back exactly.
    """
    if image is None or image.ndim < 2:
        return []
    height, width = image.shape[:2]
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox_top_left]
    except (TypeError, ValueError):
        return []
    span = max(1.0, x2 - x1, y2 - y1)
    margin = max(int(min_margin), int(round(span * margin_ratio)))
    left, top = max(0, int(x1) - margin), max(0, int(y1) - margin)
    right, bottom = min(width, int(x2) + margin), min(height, int(y2) + margin)
    if right <= left or bottom <= top:
        return []
    crop = image[top:bottom, left:right]
    original_height, original_width = crop.shape[:2]
    rotation = int(rotation_degrees) % 360
    if rotation == 90:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 270:
        crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation not in {0, 180}:
        raise ValueError("rotation_degrees must be 0, 90, 180 or 270")
    elif rotation == 180:
        crop = cv2.rotate(crop, cv2.ROTATE_180)
    scale = max(1.0, float(upscale))
    if max(crop.shape[:2]) * scale > max_side:
        scale = max_side / max(crop.shape[:2])
    if scale > 1.01:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    results = ocr.predict(crop)
    if not results:
        return []
    result = results[0]
    texts = _result_field(result, "rec_texts", [])
    scores = _result_field(result, "rec_scores", [])
    boxes = _result_field(result, "rec_boxes", [])
    items: list[dict[str, Any]] = []
    for index, text in enumerate(texts):
        value = str(text or "").strip()
        box = _bbox(boxes[index]) if index < len(boxes) else None
        if not value or box is None:
            continue
        # Convert the OCR box from the optionally rotated crop back to the
        # original full-image top-left frame.  Geometry stages never consume
        # the rotated coordinates directly.
        rx1, ry1, rx2, ry2 = [float(value) / scale for value in box]
        rotated_width = float(crop.shape[1]) / scale
        rotated_height = float(crop.shape[0]) / scale
        corners = [(rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2)]
        mapped_corners: list[tuple[float, float]] = []
        for rx, ry in corners:
            if rotation == 90:
                ox, oy = ry, float(original_width) - rx
            elif rotation == 270:
                ox, oy = float(original_height) - ry, rx
            elif rotation == 180:
                ox, oy = float(original_width) - rx, float(original_height) - ry
            else:
                ox, oy = rx, ry
            mapped_corners.append((ox, oy))
        mapped = [
            left + min(point[0] for point in mapped_corners),
            top + min(point[1] for point in mapped_corners),
            left + max(point[0] for point in mapped_corners),
            top + max(point[1] for point in mapped_corners),
        ]
        items.append(
            {
                "text": value,
                "score": float(scores[index]) if index < len(scores) else 0.0,
                "bbox": [round(value, 3) for value in mapped],
                "source": "ic_local_ocr",
                "crop_top_left": [left, top, right, bottom],
                "upscale": round(scale, 4),
                "rotation_degrees": rotation,
            }
        )
    return items
