"""Coordinate conversion helpers for the contest protocol.

Local OpenCV/YOLO/PP-OCR results use a top-left origin with y increasing
downward. The contest JSON uses a bottom-left origin with y increasing upward.
Keep conversion at the protocol boundary instead of mixing coordinate systems
inside detection and geometry algorithms.
"""

from __future__ import annotations

from numbers import Real
from typing import Mapping, Sequence


Number = int | float
PointLike = Mapping[str, Number] | Sequence[Number]
BBoxLike = Sequence[Number]


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    return float(value)


def _point_xy(point: PointLike) -> tuple[float, float]:
    if isinstance(point, Mapping):
        if "x" not in point or "y" not in point:
            raise ValueError("point mapping must contain x and y")
        return _number(point["x"], "point.x"), _number(point["y"], "point.y")
    if len(point) != 2:
        raise ValueError("point sequence must contain exactly two values")
    return _number(point[0], "point[0]"), _number(point[1], "point[1]")


def _bbox_xyxy(bbox: BBoxLike) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")
    x1, y1, x2, y2 = (
        _number(bbox[0], "bbox[0]"),
        _number(bbox[1], "bbox[1]"),
        _number(bbox[2], "bbox[2]"),
        _number(bbox[3], "bbox[3]"),
    )
    if x2 < x1 or y2 < y1:
        raise ValueError(f"bbox must satisfy x1<=x2 and y1<=y2: {bbox}")
    return x1, y1, x2, y2


def _round(value: float, decimals: int | None) -> Number:
    if decimals is None:
        return value
    rounded = round(value, decimals)
    return int(rounded) if decimals == 0 else rounded


def top_left_point_to_bottom_left(
    point: PointLike,
    image_height: Number,
    *,
    decimals: int | None = 2,
) -> dict[str, Number]:
    """Convert a point from top-left to bottom-left coordinates."""
    height = _number(image_height, "image_height")
    if height <= 0:
        raise ValueError("image_height must be positive")
    x, y = _point_xy(point)
    return {"x": _round(x, decimals), "y": _round(height - y, decimals)}


def bottom_left_point_to_top_left(
    point: PointLike,
    image_height: Number,
    *,
    decimals: int | None = 2,
) -> dict[str, Number]:
    """Convert a point from bottom-left to top-left coordinates."""
    height = _number(image_height, "image_height")
    if height <= 0:
        raise ValueError("image_height must be positive")
    x, y = _point_xy(point)
    return {"x": _round(x, decimals), "y": _round(height - y, decimals)}


def top_left_bbox_to_bottom_left(
    bbox: BBoxLike,
    image_height: Number,
    *,
    decimals: int | None = 2,
) -> list[Number]:
    """Convert [x1,y1,x2,y2] from top-left to bottom-left coordinates."""
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    height = _number(image_height, "image_height")
    if height <= 0:
        raise ValueError("image_height must be positive")
    return [
        _round(x1, decimals),
        _round(height - y2, decimals),
        _round(x2, decimals),
        _round(height - y1, decimals),
    ]


def bottom_left_bbox_to_top_left(
    bbox: BBoxLike,
    image_height: Number,
    *,
    decimals: int | None = 2,
) -> list[Number]:
    """Convert [x1,y1,x2,y2] from bottom-left to top-left coordinates."""
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    height = _number(image_height, "image_height")
    if height <= 0:
        raise ValueError("image_height must be positive")
    return [
        _round(x1, decimals),
        _round(height - y2, decimals),
        _round(x2, decimals),
        _round(height - y1, decimals),
    ]
