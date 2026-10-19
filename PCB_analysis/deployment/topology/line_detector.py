"""Detect IC-side line evidence and generate pin point candidates."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


# The contest defines the pin point semantically as the terminal/tip
# connection point.  This is an image-geometry calibration parameter, not a
# contest scoring threshold.  The public GT cases calibrate the default
# terminal offset at approximately 12 px from the refined IC body edge.
DEFAULT_PIN_TIP_OFFSET = 12.0


def _in_range(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def _add_candidate(candidates: list[dict[str, Any]], candidate: dict[str, Any], radius: float = 5.0) -> None:
    x = candidate["point_top_left"]["x"]
    y = candidate["point_top_left"]["y"]
    for existing in candidates:
        ex = existing["point_top_left"]["x"]
        ey = existing["point_top_left"]["y"]
        if math.hypot(x - ex, y - ey) <= radius:
            if candidate["evidence_score"] > existing["evidence_score"]:
                existing.update(candidate)
            return
    candidates.append(candidate)


def detect_line_segments(
    image: np.ndarray,
    bbox_top_left: list[float],
    *,
    margin: int = 32,
) -> list[list[int]]:
    """Detect mostly horizontal/vertical lines near one IC bbox."""
    x1, y1, x2, y2 = [float(value) for value in bbox_top_left]
    height, width = image.shape[:2]
    left = max(0, int(math.floor(x1 - margin)))
    top = max(0, int(math.floor(y1 - margin)))
    right = min(width, int(math.ceil(x2 + margin)))
    bottom = min(height, int(math.ceil(y2 + margin)))
    if right <= left or bottom <= top:
        return []

    crop = image[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    mask = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    combined = cv2.bitwise_or(edges, mask)

    hough = cv2.HoughLinesP(
        combined,
        1,
        np.pi / 180.0,
        threshold=5,
        minLineLength=4,
        maxLineGap=3,
    )
    segments: list[list[int]] = []
    if hough is None:
        return segments
    for raw in hough[:, 0, :]:
        ax, ay, bx, by = [int(value) for value in raw]
        ax += left
        bx += left
        ay += top
        by += top
        dx = bx - ax
        dy = by - ay
        length = math.hypot(dx, dy)
        if length < 4:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        if min(angle, abs(180.0 - angle)) <= 12.0 or abs(angle - 90.0) <= 12.0:
            segments.append([ax, ay, bx, by])
    return segments


def _horizontal_y(segment: list[int], x: float) -> float | None:
    ax, ay, bx, by = segment
    low, high = sorted((ax, bx))
    if not _in_range(x, low, high) or bx == ax:
        return None
    return ay + (by - ay) * ((x - ax) / (bx - ax))


def _vertical_x(segment: list[int], y: float) -> float | None:
    ax, ay, bx, by = segment
    low, high = sorted((ay, by))
    if not _in_range(y, low, high) or by == ay:
        return None
    return ax + (bx - ax) * ((y - ay) / (by - ay))


def _longest_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values.tolist():
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def refine_bbox_from_outline(
    image: np.ndarray,
    bbox_top_left: list[float],
    *,
    search_margin: int = 32,
) -> list[float]:
    """Tighten a detector box to the visible IC body outline when possible.

    YOLO boxes are component boxes, not guaranteed to coincide with the
    rectangle used by the schematic renderer. Pin coordinates are evaluated
    at roughly 5 px, so using a loose box can shift every row. We therefore
    look for long dark vertical/horizontal strokes close to each detector
    edge and use them only when their support is clearly stronger than local
    text/wire clutter. If no reliable outline is found, the original box is
    returned unchanged.
    """
    x1, y1, x2, y2 = [float(value) for value in bbox_top_left]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray < 100
    height, width = gray.shape[:2]
    out = [x1, y1, x2, y2]

    def best_vertical(initial: float, low: float, high: float, span: float) -> float | None:
        if span < 50.0:
            return None
        lo = max(0, int(round(initial - search_margin)))
        hi = min(width - 1, int(round(initial + search_margin)))
        y0 = max(0, int(round(low)))
        y1i = min(height, int(round(high)) + 1)
        if hi < lo or y1i <= y0:
            return None
        supports = np.asarray(
            [
                _longest_true_run(dark[y0:y1i, x])
                for x in range(lo, hi + 1)
            ],
            dtype=float,
        ) / max(1.0, span)
        valid = np.flatnonzero(supports >= 0.45)
        if valid.size == 0:
            return None
        index = min(valid.tolist(), key=lambda item: abs((lo + item) - initial))
        return float(lo + index)

    def best_horizontal(initial: float, low: float, high: float, span: float) -> float | None:
        if span < 50.0:
            return None
        lo = max(0, int(round(initial - search_margin)))
        hi = min(height - 1, int(round(initial + search_margin)))
        x0 = max(0, int(round(low)))
        x1i = min(width, int(round(high)) + 1)
        if hi < lo or x1i <= x0:
            return None
        supports = np.asarray(
            [
                _longest_true_run(dark[y, x0:x1i])
                for y in range(lo, hi + 1)
            ],
            dtype=float,
        ) / max(1.0, span)
        valid = np.flatnonzero(supports >= 0.45)
        if valid.size == 0:
            return None
        index = min(valid.tolist(), key=lambda item: abs((lo + item) - initial))
        return float(lo + index)

    left = best_vertical(x1, y1, y2, y2 - y1)
    right = best_vertical(x2, y1, y2, y2 - y1)
    top = best_horizontal(y1, x1, x2, x2 - x1)
    bottom = best_horizontal(y2, x1, x2, x2 - x1)
    if left is not None and left < x2 - 20:
        out[0] = left
    if right is not None and right > out[0] + 20:
        out[2] = right
    if top is not None and top < y2 - 20:
        out[1] = top
    if bottom is not None and bottom > out[1] + 20:
        out[3] = bottom
    return [round(value, 2) for value in out]


def _scan_side_candidates(
    image: np.ndarray,
    bbox_top_left: list[float],
    *,
    expected_offset: float,
    margin: int,
) -> list[dict[str, Any]]:
    """Find pin rows by scanning the narrow outside bands of the IC.

    Hough lines are useful for long wires but are easily confused by the IC
    outline and nearby text. A pin stub normally produces a dark horizontal or
    vertical run that reaches the corresponding bbox side, so a 1-D scan gives
    a cleaner row/column estimate for the 5 px scoring tolerance.
    """
    x1, y1, x2, y2 = [float(value) for value in bbox_top_left]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray < 210
    height, width = gray.shape[:2]
    candidates: list[dict[str, Any]] = []
    min_run = 5

    def cluster(values: list[tuple[float, float]], side: str) -> None:
        if not values:
            return
        if side in {"left", "right"}:
            values = [
                item for item in values
                if y1 + 10.0 < item[0] < y2 - 10.0
            ]
        else:
            values = [
                item for item in values
                if x1 + 10.0 < item[0] < x2 - 10.0
            ]
        if not values:
            return
        values.sort(key=lambda item: item[0])
        groups: list[list[tuple[float, float]]] = []
        for value in values:
            # A 2–4 px thick raster line can produce separate dark runs on
            # adjacent rows. Merge those into one physical pin row, while
            # keeping the usual schematic pin pitch distinct.
            if not groups or value[0] - groups[-1][-1][0] > 5.0:
                groups.append([value])
            else:
                groups[-1].append(value)
        for group in groups:
            coordinate = sum(item[0] * item[1] for item in group) / max(
                1e-9, sum(item[1] for item in group)
            )
            score = min(1.0, max(item[1] for item in group))
            if side in {"left", "right"}:
                point = {
                    "x": round(x1 - expected_offset if side == "left" else x2 + expected_offset, 2),
                    "y": round(coordinate, 2),
                }
                source = "scan_horizontal"
            else:
                point = {
                    "x": round(coordinate, 2),
                    "y": round(y1 - expected_offset if side == "top" else y2 + expected_offset, 2),
                }
                source = "scan_vertical"
            candidates.append(
                {
                    "side": side,
                    "point_top_left": point,
                    "evidence_score": round(0.78 + 0.22 * score, 4),
                    "source": source,
                }
            )

    left_values: list[tuple[float, float]] = []
    right_values: list[tuple[float, float]] = []
    top_values: list[tuple[float, float]] = []
    bottom_values: list[tuple[float, float]] = []
    y_start, y_end = max(0, int(y1)), min(height, int(y2) + 1)
    x_start, x_end = max(0, int(x1)), min(width, int(x2) + 1)

    for y in range(y_start, y_end):
        left_a, left_b = max(0, int(x1 - margin)), max(0, int(x1 - 1))
        right_a, right_b = min(width, int(x2 + 1)), min(width, int(x2 + margin) + 1)
        left_run = _longest_true_run(dark[y, left_a:left_b])
        right_run = _longest_true_run(dark[y, right_a:right_b])
        if left_run >= min_run:
            left_values.append((float(y), min(1.0, left_run / max(1.0, margin))))
        if right_run >= min_run:
            right_values.append((float(y), min(1.0, right_run / max(1.0, margin))))

    for x in range(x_start, x_end):
        top_a, top_b = max(0, int(y1 - margin)), max(0, int(y1 - 1))
        bottom_a, bottom_b = min(height, int(y2 + 1)), min(height, int(y2 + margin) + 1)
        top_run = _longest_true_run(dark[top_a:top_b, x])
        bottom_run = _longest_true_run(dark[bottom_a:bottom_b, x])
        if top_run >= min_run:
            top_values.append((float(x), min(1.0, top_run / max(1.0, margin))))
        if bottom_run >= min_run:
            bottom_values.append((float(x), min(1.0, bottom_run / max(1.0, margin))))

    cluster(left_values, "left")
    cluster(right_values, "right")
    cluster(top_values, "top")
    cluster(bottom_values, "bottom")
    return candidates


def generate_pin_candidates(
    image: np.ndarray,
    bbox_top_left: list[float],
    *,
    margin: int = 32,
    expected_offset: float = DEFAULT_PIN_TIP_OFFSET,
) -> list[dict[str, Any]]:
    """Generate pin points just outside the four sides of an IC bbox.

    The point is sampled near the expected terminal offset from the body, not
    forced onto the bbox edge. Public GT examples show pin points several
    pixels outside the component bbox.
    """
    bbox_top_left = refine_bbox_from_outline(image, bbox_top_left)
    x1, y1, x2, y2 = [float(value) for value in bbox_top_left]
    segments = detect_line_segments(image, bbox_top_left, margin=margin)
    candidates: list[dict[str, Any]] = _scan_side_candidates(
        image,
        bbox_top_left,
        expected_offset=expected_offset,
        margin=margin,
    )
    side_band = max(12.0, expected_offset + 5.0)
    # Pin centres should fall on the component span.  A wider band lets the
    # long IC outline leak into the candidate set, especially at the top and
    # bottom borders.
    y_band = 0.5
    x_band = 0.5
    vertical_inner_pad = min(6.0, max(2.0, (x2 - x1) * 0.08))

    # Hough often returns only the part of a short pin wire outside the
    # component border.  Therefore the segment does not necessarily cross
    # ``x1 - expected_offset``/``x2 + expected_offset``.  We accept a small
    # gap to the border and use the segment's local midpoint as the pin
    # ordinate/abscissa.  This is more stable than requiring an exact Hough
    # intersection with the sampling point.
    reach_gap = max(8.0, expected_offset + 8.0)

    for segment in segments:
        ax, ay, bx, by = segment
        dx, dy = bx - ax, by - ay
        angle = abs(math.degrees(math.atan2(dy, dx))) % 180.0
        length = math.hypot(dx, dy)
        is_horizontal = min(angle, abs(180.0 - angle)) <= 12.0
        is_vertical = abs(angle - 90.0) <= 12.0

        if is_horizontal:
            for side, target_x in (("left", x1 - expected_offset), ("right", x2 + expected_offset)):
                low_x, high_x = sorted((float(ax), float(bx)))
                if side == "left":
                    # A left pin must have visible ink outside the left body
                    # edge.  This rejects the IC's bottom/top outline, which
                    # often starts one pixel before x1 and was previously
                    # misclassified as a left pin.
                    if not (low_x < x1 - 2.0 and high_x <= x1 + x_band + 2.0):
                        continue
                else:
                    # Symmetric rule for the right side: the segment must
                    # extend outside x2, not merely cross the body interior.
                    if not (high_x > x2 + 2.0 and low_x >= x2 - x_band - 2.0):
                        continue
                y = _horizontal_y(segment, target_x)
                if y is None:
                    y = (float(ay) + float(by)) / 2.0
                if not _in_range(y, y1 - y_band, y2 + y_band):
                    continue
                # Text strokes and the rectangular IC outline commonly touch
                # the first/last few pixels of a side.  Real pin rows in the
                # supplied schematics are inset from those corners; keep the
                # inset rule here for Hough candidates, while the raster scan
                # still handles legitimate short wires.
                if y <= y1 + 8.0 or y >= y2 - 8.0:
                    continue
                side_edge = x1 if side == "left" else x2
                if abs(target_x - side_edge) > side_band:
                    continue
                _add_candidate(
                    candidates,
                    {
                        "side": side,
                        "point_top_left": {"x": round(target_x, 2), "y": round(y, 2)},
                        "evidence_score": round(min(1.0, length / 30.0), 4),
                        "source": "hough_horizontal",
                        "segment": segment,
                    },
                )

        if is_vertical:
            for side, target_y in (("top", y1 - expected_offset), ("bottom", y2 + expected_offset)):
                low_y, high_y = sorted((float(ay), float(by)))
                if side == "top":
                    if not (low_y < y1 - 2.0 and high_y <= y1 + y_band + 2.0):
                        continue
                else:
                    if not (high_y > y2 + 2.0 and low_y >= y2 - y_band - 2.0):
                        continue
                x = _vertical_x(segment, target_y)
                if x is None:
                    x = (float(ax) + float(bx)) / 2.0
                # Top/bottom pin wires are expected inside the horizontal
                # extent of the body.  The small interior pad rejects the
                # long left/right outline borders while retaining genuine
                # vertical terminals, including long wires.
                if not _in_range(x, x1 + vertical_inner_pad, x2 - vertical_inner_pad):
                    continue
                if x <= x1 + 8.0 or x >= x2 - 8.0:
                    continue
                side_edge = y1 if side == "top" else y2
                if abs(target_y - side_edge) > side_band:
                    continue
                _add_candidate(
                    candidates,
                    {
                        "side": side,
                        "point_top_left": {"x": round(x, 2), "y": round(target_y, 2)},
                        "evidence_score": round(min(1.0, length / 30.0), 4),
                        "source": "hough_vertical",
                        "segment": segment,
                    },
                )

    return sorted(candidates, key=lambda item: (-item["evidence_score"], item["side"]))
