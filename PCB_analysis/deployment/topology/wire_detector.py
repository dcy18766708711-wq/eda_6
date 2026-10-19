"""Extract schematic wire geometry from a raster image.

The contest scores two-point edge endpoints.  This module therefore recovers
the physical wire layer, validates morphology/Hough line candidates against a
4-neighbour skeleton, supplements only image-supported short paths, and
records component-terminal and junction candidates.  All coordinates in this
module are image-native top-left coordinates until the protocol conversion
boundary.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import cv2
import numpy as np


def _box(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, value)
    except (TypeError, ValueError):
        return None
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def _point(value: Any) -> tuple[float, float]:
    if isinstance(value, dict):
        return float(value["x"]), float(value["y"])
    return float(value[0]), float(value[1])


def _segment_points(segment: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    points = segment.get("points_top_left")
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError("wire segment must contain two points")
    return _point(points[0]), _point(points[1])


def _as_point(point: tuple[float, float]) -> dict[str, float]:
    return {"x": round(float(point[0]), 2), "y": round(float(point[1]), 2)}


def _dark_mask(image: np.ndarray, threshold: int = 210) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def _mask_ocr(mask: np.ndarray, ocr_items: Iterable[dict[str, Any]] | None) -> None:
    if not ocr_items:
        return
    height, width = mask.shape[:2]
    for item in ocr_items:
        bbox = _box(item.get("bbox_xyxy_top_left") or item.get("bbox"))
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        # Remove only the glyph interior.  A small margin prevents text
        # strokes from becoming false wires while preserving nearby wires.
        cv2.rectangle(
            mask,
            (max(0, int(x1 - 1)), max(0, int(y1 - 1))),
            (min(width - 1, int(x2 + 1)), min(height - 1, int(y2 + 1))),
            0,
            thickness=-1,
        )


def _ocr_boxes(ocr_items: Iterable[dict[str, Any]] | None) -> list[tuple[float, float, float, float]]:
    boxes = [
        _box(item.get("bbox_xyxy_top_left") or item.get("bbox"))
        for item in (ocr_items or [])
        if isinstance(item, dict)
    ]
    return [box for box in boxes if box is not None]


def _point_box_distance(point: tuple[float, float], box: tuple[float, float, float, float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _segment_ocr_overlap(
    segment: dict[str, Any],
    ocr_boxes: list[tuple[float, float, float, float]],
) -> float:
    """Return an evidence score; OCR is never used as a pre-geometry mask."""
    if not ocr_boxes:
        return 0.0
    p1, p2 = _segment_points(segment)
    length = max(1.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    samples = max(3, int(round(length / 2.0)))
    inside = 0
    for index in range(samples + 1):
        ratio = index / samples
        point = (p1[0] + ratio * (p2[0] - p1[0]), p1[1] + ratio * (p2[1] - p1[1]))
        if any(x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for x1, y1, x2, y2 in ocr_boxes):
            inside += 1
    return inside / float(samples + 1)


def _mask_component_interiors(
    mask: np.ndarray,
    component_boxes: list[tuple[float, float, float, float]],
) -> None:
    """Remove symbol interiors while preserving the external terminals."""
    height, width = mask.shape[:2]
    for x1, y1, x2, y2 in component_boxes:
        # Keep a two-pixel boundary band.  External wire/pin strokes remain
        # visible, while resistor zigzags, transistor bodies and IC text do
        # not become wire candidates.
        left, top = max(0, int(round(x1 + 2))), max(0, int(round(y1 + 2)))
        right, bottom = min(width - 1, int(round(x2 - 2))), min(height - 1, int(round(y2 - 2)))
        if right >= left and bottom >= top:
            cv2.rectangle(mask, (left, top), (right, bottom), 0, thickness=-1)


def _mask_page_frame(mask: np.ndarray) -> None:
    """Remove the large illustration border, which is not a schematic wire."""
    height, width = mask.shape[:2]
    row_support = (mask > 0).sum(axis=1)
    col_support = (mask > 0).sum(axis=0)
    for row in np.flatnonzero(row_support > 0.45 * width):
        y0, y1 = max(0, int(row) - 2), min(height, int(row) + 3)
        mask[y0:y1, :] = 0
    for col in np.flatnonzero(col_support > 0.45 * height):
        x0, x1 = max(0, int(col) - 2), min(width, int(col) + 3)
        mask[:, x0:x1] = 0


def _looks_like_body_outline(
    p1: tuple[float, float],
    p2: tuple[float, float],
    component_boxes: list[tuple[float, float, float, float]],
) -> bool:
    ax, ay = p1
    bx, by = p2
    length = math.hypot(bx - ax, by - ay)
    if length < 1:
        return True
    horizontal = abs(by - ay) <= 5.0
    vertical = abs(bx - ax) <= 5.0
    for x1, y1, x2, y2 in component_boxes:
        if horizontal:
            near_top = abs((ay + by) / 2.0 - y1) <= 5.0
            near_bottom = abs((ay + by) / 2.0 - y2) <= 5.0
            overlap = max(0.0, min(max(ax, bx), x2) - max(min(ax, bx), x1))
            segment_overlap = max(0.0, min(max(ax, bx), x2) - max(min(ax, bx), x1))
            segment_length = max(1.0, abs(bx - ax))
            # A symbol outline may be fragmented by OCR or by another
            # detector, so checking only the fraction of the whole bbox is
            # too weak.  Reject a line that is almost entirely inside the
            # component and lies on its top/bottom boundary.
            if (
                (near_top or near_bottom)
                and segment_overlap / segment_length > 0.85
                and overlap / max(1.0, x2 - x1) > 0.12
            ):
                return True
        if vertical:
            near_left = abs((ax + bx) / 2.0 - x1) <= 5.0
            near_right = abs((ax + bx) / 2.0 - x2) <= 5.0
            overlap = max(0.0, min(max(ay, by), y2) - max(min(ay, by), y1))
            segment_overlap = max(0.0, min(max(ay, by), y2) - max(min(ay, by), y1))
            segment_length = max(1.0, abs(by - ay))
            if (
                (near_left or near_right)
                and segment_overlap / segment_length > 0.85
                and overlap / max(1.0, y2 - y1) > 0.12
            ):
                return True
    return False


def _inside_ocr(
    p1: tuple[float, float],
    p2: tuple[float, float],
    ocr_boxes: list[tuple[float, float, float, float]],
) -> bool:
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    return any(x1 <= mx <= x2 and y1 <= my <= y2 for x1, y1, x2, y2 in ocr_boxes)


def _raw_segments(
    mask: np.ndarray,
    component_boxes: list[tuple[float, float, float, float]],
    ocr_boxes: list[tuple[float, float, float, float]],
    *,
    min_line_length: int,
    stroke_width: float = 1.0,
) -> list[tuple[tuple[float, float], tuple[float, float], float]]:
    height, width = mask.shape[:2]
    # Opening length follows the measured raster stroke rather than the
    # absolute image dimensions.  A fixed image-ratio kernel becomes too
    # large on high-resolution sheets and erases short real wires.
    kernel_length = max(3, min(31, int(round(3.0 * max(1.0, float(stroke_width)) + 2.0))))
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    )
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    )
    line_mask = cv2.bitwise_or(horizontal, vertical)
    hough = cv2.HoughLinesP(
        line_mask,
        1,
        np.pi / 180.0,
        threshold=max(5, min_line_length // 2),
        minLineLength=min_line_length,
        maxLineGap=4,
    )
    if hough is None:
        return []
    result: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    endpoint_reach = max(2, min(8, kernel_length // 2 + 2))
    height, width = mask.shape[:2]

    def extend_to_raw_mask(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        horizontal = abs(second[0] - first[0]) >= abs(second[1] - first[1])
        if horizontal:
            axis = (first[1] + second[1]) / 2.0
            start, end = sorted((first[0], second[0]))
            lateral = max(1, min(3, int(round(max(1.0, stroke_width)))))
            for direction in (-1, 1):
                for step in range(1, endpoint_reach + 1):
                    x = int(round(start - step)) if direction < 0 else int(round(end + step))
                    y = int(round(axis))
                    if not (0 <= x < width and 0 <= y < height):
                        break
                    supported = any(
                        0 <= y + offset < height and mask[y + offset, x]
                        for offset in range(-lateral, lateral + 1)
                    )
                    if not supported:
                        break
                    if direction < 0:
                        start = float(x)
                    else:
                        end = float(x)
            return (start, axis), (end, axis)
        axis = (first[0] + second[0]) / 2.0
        start, end = sorted((first[1], second[1]))
        lateral = max(1, min(3, int(round(max(1.0, float(stroke_width))))))
        for direction in (-1, 1):
            for step in range(1, endpoint_reach + 1):
                y = int(round(start - step)) if direction < 0 else int(round(end + step))
                x = int(round(axis))
                if not (0 <= x < width and 0 <= y < height):
                    break
                supported = any(
                    0 <= x + offset < width and mask[y, x + offset]
                    for offset in range(-lateral, lateral + 1)
                )
                if not supported:
                    break
                if direction < 0:
                    start = float(y)
                else:
                    end = float(y)
        return (axis, start), (axis, end)

    for raw in hough[:, 0, :]:
        ax, ay, bx, by = map(float, raw)
        p1, p2 = (ax, ay), (bx, by)
        length = math.hypot(bx - ax, by - ay)
        if length < min_line_length:
            continue
        angle = abs(math.degrees(math.atan2(by - ay, bx - ax))) % 180.0
        horizontal_line = min(angle, abs(180.0 - angle)) <= 8.0
        vertical_line = abs(angle - 90.0) <= 8.0
        if not (horizontal_line or vertical_line):
            continue
        # OCR is posterior evidence only.  A text box may overlap a real
        # short wire or a pin transition, so it must never delete a geometric
        # candidate before skeleton/support validation.
        if _looks_like_body_outline(p1, p2, component_boxes):
            continue
        p1, p2 = extend_to_raw_mask(p1, p2)
        result.append((p1, p2, min(1.0, length / 80.0)))
    # A morphology opening is appropriate for the main long-line channel,
    # but it can erase a real 2--5 px interval at a pin or junction.  Add a
    # second, short-line observation channel directly from the measured raw
    # mask.  It is deliberately not accepted as geometry on its own: the
    # downstream skeleton-support gate, terminal/junction evidence, and
    # candidate deduplication remain authoritative.
    short_length = max(2, min(8, int(round(1.2 * max(1.0, float(stroke_width))))))
    short_gap = int(round(max(1.0, float(stroke_width))))
    short_gap = max(1, min(3, short_gap))
    short_hough = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180.0,
        threshold=max(2, int(round(1.2 * max(1.0, float(stroke_width))))),
        minLineLength=short_length,
        maxLineGap=short_gap,
    )
    if short_hough is not None:
        for raw in short_hough[:, 0, :]:
            ax, ay, bx, by = map(float, raw)
            p1, p2 = (ax, ay), (bx, by)
            length = math.hypot(bx - ax, by - ay)
            if length < 0.75:
                continue
            angle = abs(math.degrees(math.atan2(by - ay, bx - ax))) % 180.0
            horizontal_line = min(angle, abs(180.0 - angle)) <= 8.0
            vertical_line = abs(angle - 90.0) <= 8.0
            if not (horizontal_line or vertical_line):
                continue
            if _looks_like_body_outline(p1, p2, component_boxes):
                continue
            result.append((p1, p2, min(0.75, length / 80.0)))
    return result


def _merge_axis_segments(
    raw: list[tuple[tuple[float, float], tuple[float, float], float]],
    *,
    gap_tolerance: float = 6.0,
    support_mask: np.ndarray | None = None,
    transition_mask: np.ndarray | None = None,
    transition_anchor_points: Iterable[tuple[float, float]] | None = None,
    transition_anchor_radius: float = 0.0,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for p1, p2, confidence in raw:
        ax, ay = p1
        bx, by = p2
        if abs(by - ay) <= abs(bx - ax):
            orientation = "horizontal"
            coordinate = (ay + by) / 2.0
            start, end = sorted((ax, bx))
        else:
            orientation = "vertical"
            coordinate = (ax + bx) / 2.0
            start, end = sorted((ay, by))
        matched = None
        for group in groups:
            if group["orientation"] != orientation:
                continue
            if abs(group["coordinate"] - coordinate) > 3.0:
                continue
            separated = start > group["end"] + gap_tolerance or end < group["start"] - gap_tolerance
            if separated:
                continue
            # Even a short merge gap must be supported by the measured mask.
            # Otherwise a Hough fragment from a nearby glyph/parallel track
            # can be joined merely because its endpoints are close.  The
            # check is applied only when there is a real positive gap; true
            # overlaps remain mergeable.
            short_gap = (
                start - group["end"]
                if start >= group["end"]
                else group["start"] - end
            )
            if short_gap > 0.5 and support_mask is not None:
                if orientation == "horizontal":
                    left = int(round(min(group["end"], start)))
                    right = int(round(max(group["end"], start)))
                    y = int(round((group["coordinate"] + coordinate) / 2.0))
                    samples = [(x, y) for x in range(left, right + 1)]
                else:
                    top = int(round(min(group["end"], start)))
                    bottom = int(round(max(group["end"], start)))
                    x = int(round((group["coordinate"] + coordinate) / 2.0))
                    samples = [(x, y) for y in range(top, bottom + 1)]
                height, width = support_mask.shape[:2]
                supported = 0
                for sx, sy in samples:
                    x0, x1 = max(0, sx - 1), min(width, sx + 2)
                    y0, y1 = max(0, sy - 1), min(height, sy + 2)
                    supported += int(cv2.countNonZero(support_mask[y0:y1, x0:x1]) > 0)
                if not samples or supported / float(len(samples)) < 0.5:
                    continue
            if start > group["end"] + 6.0 or end < group["start"] - 6.0:
                # A larger-than-default gap is allowed only when the
                # measured image still contains local ink on the same track.
                # This recovers a green run interrupted by a red terminal or
                # antialiasing, but never bridges an unsupported blank gap.
                if support_mask is None:
                    continue
                if orientation == "horizontal":
                    left = int(round(group["end"]))
                    right = int(round(start))
                    y = int(round((group["coordinate"] + coordinate) / 2.0))
                    samples = [
                        (x, y)
                        for x in range(min(left, right), max(left, right) + 1)
                    ]
                else:
                    top = int(round(group["end"]))
                    bottom = int(round(start))
                    x = int(round((group["coordinate"] + coordinate) / 2.0))
                    samples = [
                        (x, y)
                        for y in range(min(top, bottom), max(top, bottom) + 1)
                    ]
                height, width = support_mask.shape[:2]
                supported = 0
                transition_supported = 0
                for sx, sy in samples:
                    x0, x1 = max(0, sx - 1), min(width, sx + 2)
                    y0, y1 = max(0, sy - 1), min(height, sy + 2)
                    supported += int(cv2.countNonZero(support_mask[y0:y1, x0:x1]) > 0)
                    if transition_mask is not None:
                        transition_supported += int(
                            cv2.countNonZero(transition_mask[y0:y1, x0:x1]) > 0
                        )
                support_ratio = supported / float(len(samples)) if samples else 0.0
                transition_ratio = (
                    transition_supported / float(len(samples)) if samples else 0.0
                )
                middle = samples[len(samples) // 2] if samples else (0, 0)
                near_transition_anchor = (
                    any(
                        math.hypot(float(middle[0]) - float(px), float(middle[1]) - float(py))
                        <= max(0.0, float(transition_anchor_radius))
                        for px, py in (transition_anchor_points or [])
                    )
                    if transition_anchor_points is not None
                    else True
                )
                if (
                    not samples
                    or support_ratio < 0.35
                    or (transition_mask is not None and transition_ratio < 0.2)
                    or (transition_mask is not None and not near_transition_anchor)
                ):
                    continue
            matched = group
            break
        if matched is None:
            groups.append(
                {
                    "orientation": orientation,
                    "coordinate": coordinate,
                    "start": start,
                    "end": end,
                    "confidence": confidence,
                }
            )
        else:
            weight = max(1.0, matched["end"] - matched["start"])
            new_weight = max(1.0, end - start)
            matched["coordinate"] = (matched["coordinate"] * weight + coordinate * new_weight) / (weight + new_weight)
            matched["start"] = min(matched["start"], start)
            matched["end"] = max(matched["end"], end)
            matched["confidence"] = max(matched["confidence"], confidence)
    segments: list[dict[str, Any]] = []
    for group in groups:
        if group["orientation"] == "horizontal":
            points = [
                {"x": round(group["start"], 2), "y": round(group["coordinate"], 2)},
                {"x": round(group["end"], 2), "y": round(group["coordinate"], 2)},
            ]
        else:
            points = [
                {"x": round(group["coordinate"], 2), "y": round(group["start"], 2)},
                {"x": round(group["coordinate"], 2), "y": round(group["end"], 2)},
            ]
        segments.append(
            {
                "points_top_left": points,
                "orientation": group["orientation"],
                "source": "morphology_hough_merged",
                "confidence": round(float(group["confidence"]), 4),
            }
        )
    return segments


def _merge_local_red_gap_segments(
    segments: list[dict[str, Any]],
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    *,
    stroke_width: float,
    transition_anchor_points: Iterable[tuple[float, float]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Merge only a short red-rendered interruption on one green wire track.

    The red pixels are inspected in a bounded gap between two already
    detected green Hough segments.  Acceptance requires green support on both
    gap sides, red support in the middle, and high fused support throughout.
    Thus this is a local terminal transition, not a red global wire layer.
    """
    if not segments or red_mask is None or not cv2.countNonZero(red_mask):
        return segments, 0
    height, width = green_mask.shape[:2]
    max_gap = max(4.0, min(24.0, 5.0 * max(1.0, float(stroke_width))))
    axis_tolerance = max(2.0, min(5.0, 1.8 * max(1.0, float(stroke_width))))
    green = green_mask > 0
    red = red_mask > 0
    result = [dict(item) for item in segments]
    merged_count = 0

    def interval(item: dict[str, Any]) -> tuple[str, float, float, float] | None:
        try:
            first, second = _segment_points(item)
        except (KeyError, TypeError, ValueError):
            return None
        if item.get("orientation") == "horizontal" or abs(first[0] - second[0]) >= abs(first[1] - second[1]):
            return "horizontal", (first[1] + second[1]) / 2.0, min(first[0], second[0]), max(first[0], second[0])
        return "vertical", (first[0] + second[0]) / 2.0, min(first[1], second[1]), max(first[1], second[1])

    # A bounded number of passes handles adjacent local gaps.  Rebuild the
    # ordered list on every pass; using cached list indices here is unsafe
    # because merging deletes one of the two fragments.
    # Keep the bounded pass count so a dense high-resolution page cannot turn
    # local gap repair into an expensive global merge loop.
    for _pass in range(3):
        ordered: list[tuple[int, str, float, float, float]] = []
        for index, item in enumerate(result):
            info = interval(item)
            if info is None:
                continue
            orientation, coordinate, start, end = info
            ordered.append((index, orientation, coordinate, start, end))
        ordered.sort(key=lambda value: (value[1], value[2], value[3], value[4]))
        changed = False
        for left_pos, right_pos in zip(range(len(ordered)), range(1, len(ordered))):
            left_index, left_orientation, left_coordinate, left_start, left_end = ordered[left_pos]
            right_index, right_orientation, right_coordinate, right_start, right_end = ordered[right_pos]
            if left_orientation != right_orientation:
                continue
            if abs(left_coordinate - right_coordinate) > axis_tolerance:
                continue
            gap = right_start - left_end
            if gap <= 0.5 or gap > max_gap:
                continue
            orientation = left_orientation
            coordinate = (left_coordinate + right_coordinate) / 2.0
            if transition_anchor_points is not None:
                gap_midpoint = (
                    ((left_end + right_start) / 2.0, coordinate)
                    if orientation == "horizontal"
                    else (coordinate, (left_end + right_start) / 2.0)
                )
                anchor_radius = max(3.0, min(12.0, 2.0 * max(1.0, float(stroke_width)) + 0.5 * gap))
                if not any(
                    math.hypot(gap_midpoint[0] - float(point[0]), gap_midpoint[1] - float(point[1]))
                    <= anchor_radius
                    for point in transition_anchor_points
                ):
                    continue
            samples = max(3, int(round(gap * 2.0)))
            fused_hits = 0
            red_hits = 0
            for sample_index in range(samples + 1):
                ratio = sample_index / float(samples)
                if orientation == "horizontal":
                    sx, sy = int(round(left_end + ratio * gap)), int(round(coordinate))
                else:
                    sx, sy = int(round(coordinate)), int(round(left_end + ratio * gap))
                x0, x1 = max(0, sx - 1), min(width, sx + 2)
                y0, y1 = max(0, sy - 1), min(height, sy + 2)
                green_hit = bool(np.any(green[y0:y1, x0:x1]))
                red_hit = bool(np.any(red[y0:y1, x0:x1]))
                fused_hits += int(green_hit or red_hit)
                red_hits += int(red_hit)
            # Require green on the outside of both ends; the gap itself may
            # be entirely red at the rendered terminal.
            side_span = max(2, int(math.ceil(2.0 * max(1.0, float(stroke_width)))))
            green_left = 0
            green_right = 0
            side_samples = side_span + 1
            for side_index in range(side_samples):
                left_axis = int(round(left_end - side_span + side_index))
                right_axis = int(round(right_start + side_index))
                if orientation == "horizontal":
                    side_points = ((left_axis, int(round(coordinate)), "left"), (right_axis, int(round(coordinate)), "right"))
                else:
                    side_points = ((int(round(coordinate)), left_axis, "left"), (int(round(coordinate)), right_axis, "right"))
                for sx, sy, side_name in side_points:
                    x0, x1 = max(0, sx - 1), min(width, sx + 2)
                    y0, y1 = max(0, sy - 1), min(height, sy + 2)
                    if np.any(green[y0:y1, x0:x1]):
                        if side_name == "left":
                            green_left += 1
                        else:
                            green_right += 1
            if (
                fused_hits / float(samples + 1) < 0.7
                or red_hits < max(1, int(round(0.2 * (samples + 1))))
                or green_left < max(1, int(round(0.5 * side_samples)))
                or green_right < max(1, int(round(0.5 * side_samples)))
            ):
                continue
            merged = dict(result[left_index])
            left_points = _segment_points(result[left_index])
            right_points = _segment_points(result[right_index])
            if orientation == "horizontal":
                merged["points_top_left"] = [
                    _as_point((min(left_points[0][0], left_points[1][0]), coordinate)),
                    _as_point((max(right_points[0][0], right_points[1][0]), coordinate)),
                ]
            else:
                merged["points_top_left"] = [
                    _as_point((coordinate, min(left_points[0][1], left_points[1][1]))),
                    _as_point((coordinate, max(right_points[0][1], right_points[1][1]))),
                ]
            merged["source"] = "wire_hough_local_red_gap"
            merged["local_red_gap_evidence"] = True
            result[left_index] = merged
            del result[right_index]
            merged_count += 1
            changed = True
            break
        if not changed:
            break
    return result, merged_count


def _red_gap_track_union_candidates(
    segments: list[dict[str, Any]],
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    transition_anchor_points: Iterable[tuple[float, float]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Add a physical track hypothesis across a validated red terminal gap.

    Some schematic renderers draw a component terminal in red over a green
    wire.  The green Hough channel consequently produces two intervals, even
    though the visible circuit contains one collinear track.  This helper
    adds one *supplementary* union candidate only when all of the following
    are observed:

    * both sides are existing Hough candidates on the same adaptive axis;
    * the gap is short relative to the measured stroke and contains red ink;
    * green support exists immediately on both sides; and
    * a previously validated terminal/junction anchor is close to the gap.

    It never creates a wire from a bbox, a pin-library order, an edge count,
    or red pixels alone.  The original pieces remain available for genuine
    graph splits; the union is a recall supplement for the physical track.
    """
    if not segments or red_mask is None or not cv2.countNonZero(red_mask):
        return []
    if green_mask.shape != red_mask.shape:
        return []
    height, width = green_mask.shape[:2]
    stroke = max(1.0, float(stroke_width))
    axis_tolerance = max(2.0, min(5.0, 1.5 * stroke))
    max_gap = max(4.0, min(16.0, 5.0 * stroke))
    green = green_mask > 0
    red = red_mask > 0
    anchors = [(float(x), float(y)) for x, y in transition_anchor_points]
    tracks: list[dict[str, Any]] = []
    for segment in segments:
        if "hough" not in str(segment.get("source") or ""):
            continue
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        orientation = str(segment.get("orientation") or "")
        if orientation not in {"horizontal", "vertical"}:
            continue
        coordinate = (
            (first[1] + second[1]) / 2.0
            if orientation == "horizontal"
            else (first[0] + second[0]) / 2.0
        )
        start, end = sorted(
            (first[0], second[0])
            if orientation == "horizontal"
            else (first[1], second[1])
        )
        track = next(
            (
                item
                for item in tracks
                if item["orientation"] == orientation
                and abs(item["coordinate"] - coordinate) <= axis_tolerance
            ),
            None,
        )
        if track is None:
            track = {"orientation": orientation, "coordinate": coordinate, "intervals": []}
            tracks.append(track)
        intervals = track["intervals"]
        intervals.append((start, end))
        count = len(intervals)
        track["coordinate"] = (
            track["coordinate"] * (count - 1) + coordinate
        ) / float(count)

    def gap_is_valid(
        orientation: str,
        coordinate: float,
        left_end: float,
        right_start: float,
    ) -> bool:
        gap = right_start - left_end
        if gap <= 0.5 or gap > max_gap:
            return False
        samples = max(3, int(round(2.0 * gap)))
        red_hits = 0
        for index in range(samples + 1):
            value = left_end + gap * index / float(samples)
            x = int(round(value)) if orientation == "horizontal" else int(round(coordinate))
            y = int(round(coordinate)) if orientation == "horizontal" else int(round(value))
            x0, x1 = max(0, x - 1), min(width, x + 2)
            y0, y1 = max(0, y - 1), min(height, y + 2)
            red_hits += int(bool(np.any(red[y0:y1, x0:x1])))

        side_span = max(2, int(round(2.0 * stroke)))
        green_left = 0
        green_right = 0
        for index in range(side_span + 1):
            for value, side_name in (
                (left_end - side_span + index, "left"),
                (right_start + index, "right"),
            ):
                x = int(round(value)) if orientation == "horizontal" else int(round(coordinate))
                y = int(round(coordinate)) if orientation == "horizontal" else int(round(value))
                x0, x1 = max(0, x - 1), min(width, x + 2)
                y0, y1 = max(0, y - 1), min(height, y + 2)
                if np.any(green[y0:y1, x0:x1]):
                    if side_name == "left":
                        green_left += 1
                    else:
                        green_right += 1

        midpoint = (left_end + right_start) / 2.0
        anchor_radius = max(4.0, 2.0 * stroke + 0.5 * gap)
        near_anchor = any(
            math.hypot(
                (x - midpoint) if orientation == "horizontal" else (x - coordinate),
                (y - coordinate) if orientation == "horizontal" else (y - midpoint),
            )
            <= anchor_radius
            for x, y in anchors
        )
        return (
            red_hits >= max(1, int(round(0.2 * (samples + 1))))
            and green_left >= max(1, int(round(0.5 * (side_span + 1))))
            and green_right >= max(1, int(round(0.5 * (side_span + 1))))
            and near_anchor
        )

    def touching_join_is_valid(
        orientation: str,
        coordinate: float,
        join: float,
    ) -> bool:
        """Validate a Hough split whose intervals touch at a red terminal."""
        radius = max(3, int(round(3.0 * stroke)))
        x_center = int(round(join)) if orientation == "horizontal" else int(round(coordinate))
        y_center = int(round(coordinate)) if orientation == "horizontal" else int(round(join))
        x0, x1 = max(0, x_center - radius), min(width, x_center + radius + 1)
        y0, y1 = max(0, y_center - radius), min(height, y_center + radius + 1)
        if not np.any(red[y0:y1, x0:x1]):
            return False
        return any(
            math.hypot(x - x_center, y - y_center) <= max(4.0, 2.0 * stroke + radius)
            for x, y in anchors
        )

    def overlap_join_is_valid(
        orientation: str,
        coordinate: float,
        left_start: float,
        left_end: float,
        right_start: float,
        right_end: float,
    ) -> bool:
        """Find red terminal evidence inside overlapping Hough intervals."""
        overlap_start = max(left_start, right_start)
        overlap_end = min(left_end, right_end)
        if overlap_end < overlap_start:
            return False
        radius = max(3, int(round(3.0 * stroke)))
        axis_values = range(int(math.floor(overlap_start)), int(math.ceil(overlap_end)) + 1)
        for value in axis_values:
            x = int(round(value)) if orientation == "horizontal" else int(round(coordinate))
            y = int(round(coordinate)) if orientation == "horizontal" else int(round(value))
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            if not np.any(red[y0:y1, x0:x1]):
                continue
            # A red body outline without green tracks on both sides is not a
            # valid global union witness.
            side_span = max(2, int(round(2.0 * stroke)))
            left_support = 0
            right_support = 0
            for step in range(1, side_span + 1):
                values = (value - step, value + step)
                for side_value, side_name in zip(values, ("left", "right")):
                    sx = int(round(side_value)) if orientation == "horizontal" else int(round(coordinate))
                    sy = int(round(coordinate)) if orientation == "horizontal" else int(round(side_value))
                    sx0, sx1 = max(0, sx - 1), min(width, sx + 2)
                    sy0, sy1 = max(0, sy - 1), min(height, sy + 2)
                    if np.any(green[sy0:sy1, sx0:sx1]):
                        if side_name == "left":
                            left_support += 1
                        else:
                            right_support += 1
            if left_support < max(1, side_span // 2) or right_support < max(1, side_span // 2):
                continue
            near_anchor = any(
                math.hypot(
                    (anchor_x - value) if orientation == "horizontal" else (anchor_x - coordinate),
                    (anchor_y - coordinate) if orientation == "horizontal" else (anchor_y - value),
                ) <= max(4.0, 2.0 * stroke + radius)
                for anchor_x, anchor_y in anchors
            )
            if near_anchor:
                return True
        return False

    output: list[dict[str, Any]] = []
    for track in tracks:
        orientation = track["orientation"]
        coordinate = float(track["coordinate"])
        intervals = sorted(track["intervals"], key=lambda value: (value[0], value[1]))
        groups: list[tuple[list[tuple[float, float]], bool]] = []
        current: list[tuple[float, float]] = []
        has_valid_gap = False
        for interval in intervals:
            if not current:
                current = [interval]
                continue
            left_end = max(item[1] for item in current)
            if interval[0] <= left_end + 0.5:
                left_start = min(item[0] for item in current)
                if overlap_join_is_valid(
                    orientation,
                    coordinate,
                    left_start,
                    left_end,
                    interval[0],
                    interval[1],
                ):
                    has_valid_gap = True
                current.append(interval)
                if touching_join_is_valid(orientation, coordinate, interval[0]):
                    has_valid_gap = True
                continue
            if gap_is_valid(orientation, coordinate, left_end, interval[0]):
                current.append(interval)
                has_valid_gap = True
                continue
            groups.append((current, has_valid_gap))
            current = [interval]
            has_valid_gap = False
        if current:
            groups.append((current, has_valid_gap))

        for group, has_valid_gap in groups:
            if not has_valid_gap:
                continue
            start = min(item[0] for item in group)
            end = max(item[1] for item in group)
            if end - start < 0.75:
                continue
            if orientation == "horizontal":
                points = [_as_point((start, coordinate)), _as_point((end, coordinate))]
            else:
                points = [_as_point((coordinate, start)), _as_point((coordinate, end))]
            output.append(
                {
                    "points_top_left": points,
                    "orientation": orientation,
                    "source": "wire_red_gap_track_union",
                    "confidence": 1.0,
                    "red_gap_track_evidence": True,
                    "local_fused_layers": ["green_hough", "red_terminal", "terminal_anchor"],
                }
            )
    return output


def _red_terminal_parent_track_candidates(
    segments: list[dict[str, Any]],
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    terminal_records: Iterable[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Keep an Hough parent track when a red terminal lies inside it.

    A renderer may draw a green net continuously underneath a red pin
    terminal.  Hough then measures one long supported interval, but a later
    graph split at the terminal can discard that parent interval.  This
    helper retains the parent only when the same measured segment contains a
    validated terminal, red pixels occur around that terminal, and green
    pixels are observed on both sides along the segment axis.

    It never creates a segment from a terminal coordinate: the parent must
    already be a global Hough candidate supported by the wire mask.
    """
    if not segments or red_mask is None or green_mask.shape != red_mask.shape:
        return []
    green = green_mask > 0
    red = red_mask > 0
    stroke = max(1.0, float(stroke_width))
    radius = max(3, int(round(3.0 * stroke)))
    axis_tolerance = max(2.0, min(5.0, 1.5 * stroke))
    records = [record for record in terminal_records if record.get("pin_anchor_evidence", False)]
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()
    height, width = green.shape[:2]

    for segment in segments:
        if "hough" not in str(segment.get("source") or ""):
            continue
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        orientation = str(segment.get("orientation") or "")
        if orientation not in {"horizontal", "vertical"}:
            continue
        if orientation == "horizontal":
            coordinate = (first[1] + second[1]) / 2.0
            start, end = sorted((first[0], second[0]))
        else:
            coordinate = (first[0] + second[0]) / 2.0
            start, end = sorted((first[1], second[1]))
        if end - start < max(5.0, 4.0 * stroke):
            continue

        for record in records:
            point = record.get("skeleton_point_top_left") or record.get("point_top_left")
            if not isinstance(point, dict):
                continue
            try:
                px, py = float(point["x"]), float(point["y"])
            except (KeyError, TypeError, ValueError):
                continue
            axis_value = py if orientation == "horizontal" else px
            along = px if orientation == "horizontal" else py
            if abs(axis_value - coordinate) > axis_tolerance:
                continue
            if along <= start + axis_tolerance or along >= end - axis_tolerance:
                continue

            cx, cy = int(round(px)), int(round(py))
            x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
            y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
            if not np.any(red[y0:y1, x0:x1]):
                continue

            side_span = max(2, int(round(2.0 * stroke)))
            left_support = 0
            right_support = 0
            for step in range(1, side_span + 1):
                values = ((along - step, along + step) if orientation == "horizontal" else (along - step, along + step))
                for value, side in zip(values, ("left", "right")):
                    xx = int(round(value)) if orientation == "horizontal" else cx
                    yy = cy if orientation == "horizontal" else int(round(value))
                    sx0, sx1 = max(0, xx - 1), min(width, xx + 2)
                    sy0, sy1 = max(0, yy - 1), min(height, yy + 2)
                    if np.any(green[sy0:sy1, sx0:sx1]):
                        if side == "left":
                            left_support += 1
                        else:
                            right_support += 1
            if left_support < max(1, side_span // 2) or right_support < max(1, side_span // 2):
                continue

            key = (orientation, int(round(coordinate)), int(round(start)), int(round(end)))
            if key in seen:
                continue
            seen.add(key)
            item = dict(segment)
            item["source"] = "wire_red_terminal_parent_track"
            item["red_terminal_parent_evidence"] = True
            item["local_fused_layers"] = ["green_hough", "red_terminal", "validated_pin_anchor"]
            output.append(item)
            break
    return output


def _observed_node_parent_track_candidates(
    segments: list[dict[str, Any]],
    green_mask: np.ndarray,
    terminal_records: Iterable[dict[str, Any]],
    junction_records: Iterable[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Retain image-supported Hough parent tracks across measured graph nodes.

    The protocol can contain both a long physical run and child edges split at
    a pin/junction.  Hough already measures the parent run, so preserving it
    is safe only when a real terminal or classified junction lies inside that
    existing run and the raster has wire support on both sides.  This is
    independent of colour and does not turn a node coordinate into a new
    segment.
    """
    if not segments or green_mask.size == 0:
        return []
    green = green_mask > 0
    stroke = max(1.0, float(stroke_width))
    axis_tolerance = max(2.0, min(5.0, 1.5 * stroke))
    side_span = max(2, int(round(2.0 * stroke)))
    records = [
        record
        for record in [*terminal_records, *junction_records]
        if isinstance(record, dict)
        and (
            record.get("pin_anchor_evidence", False)
            or record.get("classification") in {"junction_candidate", "branch_candidate", "crossing_candidate"}
            or (
                isinstance(record.get("skeleton_point_top_left"), dict)
                and record.get("side") in {"left", "right", "top", "bottom"}
            )
        )
    ]
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()

    def local_green(x: int, y: int) -> bool:
        height, width = green.shape[:2]
        x0, x1 = max(0, x - 1), min(width, x + 2)
        y0, y1 = max(0, y - 1), min(height, y + 2)
        return bool(np.any(green[y0:y1, x0:x1]))

    for segment in segments:
        if "hough" not in str(segment.get("source") or ""):
            continue
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        orientation = str(segment.get("orientation") or "")
        if orientation not in {"horizontal", "vertical"}:
            continue
        if orientation == "horizontal":
            coordinate = (first[1] + second[1]) / 2.0
            start, end = sorted((first[0], second[0]))
        else:
            coordinate = (first[0] + second[0]) / 2.0
            start, end = sorted((first[1], second[1]))
        if end - start < max(8.0, 5.0 * stroke):
            continue
        for record in records:
            point = record.get("skeleton_point_top_left") or record.get("point_top_left")
            if not isinstance(point, dict):
                continue
            try:
                px, py = float(point["x"]), float(point["y"])
            except (KeyError, TypeError, ValueError):
                continue
            axis_value = py if orientation == "horizontal" else px
            along = px if orientation == "horizontal" else py
            if abs(axis_value - coordinate) > axis_tolerance:
                continue
            if along <= start + axis_tolerance or along >= end - axis_tolerance:
                continue
            left_support = 0
            right_support = 0
            for step in range(1, side_span + 1):
                values = (along - step, along + step)
                for value, side in zip(values, ("left", "right")):
                    x = int(round(value)) if orientation == "horizontal" else int(round(coordinate))
                    y = int(round(coordinate)) if orientation == "horizontal" else int(round(value))
                    if local_green(x, y):
                        if side == "left":
                            left_support += 1
                        else:
                            right_support += 1
            if left_support < max(1, side_span // 2) or right_support < max(1, side_span // 2):
                continue
            key = (orientation, int(round(coordinate)), int(round(start)), int(round(end)))
            if key in seen:
                continue
            seen.add(key)
            item = dict(segment)
            item["source"] = "wire_observed_node_parent_track"
            item["observed_node_parent_evidence"] = True
            item["local_fused_layers"] = ["green_hough", "observed_terminal_or_junction"]
            output.append(item)
            break
    return output


def _split_at_crossings(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[list[tuple[float, float]]] = []
    for segment in segments:
        p1, p2 = _segment_points(segment)
        points.append([p1, p2])
    for i, first in enumerate(segments):
        a1, a2 = _segment_points(first)
        for j in range(i + 1, len(segments)):
            second = segments[j]
            b1, b2 = _segment_points(second)
            if first["orientation"] == second["orientation"]:
                continue
            horizontal = first if first["orientation"] == "horizontal" else second
            vertical = second if first["orientation"] == "horizontal" else first
            h1, h2 = _segment_points(horizontal)
            v1, v2 = _segment_points(vertical)
            x = (v1[0] + v2[0]) / 2.0
            y = (h1[1] + h2[1]) / 2.0
            if min(h1[0], h2[0]) - 2 <= x <= max(h1[0], h2[0]) + 2 and min(v1[1], v2[1]) - 2 <= y <= max(v1[1], v2[1]) + 2:
                points[i].append((x, y))
                points[j].append((x, y))
    split: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        p1, p2 = _segment_points(segment)
        ordered = points[index]
        if segment["orientation"] == "horizontal":
            ordered.sort(key=lambda point: point[0])
        else:
            ordered.sort(key=lambda point: point[1])
        for left, right in zip(ordered, ordered[1:]):
            if math.hypot(right[0] - left[0], right[1] - left[1]) < 3.0:
                continue
            split.append(
                {
                    "points_top_left": [
                        {"x": round(left[0], 2), "y": round(left[1], 2)},
                        {"x": round(right[0], 2), "y": round(right[1], 2)},
                    ],
                    "orientation": segment["orientation"],
                    "source": segment["source"],
                    "confidence": segment["confidence"],
                }
            )
    return _merge_split_segments(split)


def _split_hough_at_local_color_transitions(
    image: np.ndarray,
    segments: list[dict[str, Any]],
    terminal_records: Iterable[dict[str, Any]],
    *,
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    stroke_width: float,
) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    """Split measured long tracks at a local red/green transition.

    The global wire mask deliberately contains only the green wire layer.  In
    the rendered schematic a terminal can nevertheless contain a short red or
    antialiased interval *inside* an otherwise continuous green track.  The
    local ROI stage observes that interval, while this helper transfers only
    its measured centre to the already supported Hough track.  It does not
    create a new segment and never promotes a red component outline globally.

    A transition is accepted only if:

    * red pixels form a short run with green support on both sides;
    * the run lies on a horizontal/vertical Hough track; and
    * the run is in the outward corridor of an observed terminal record.

    All distances are derived from the measured raster stroke and the source
    component scale carried by the terminal record.
    """
    if (
        image.ndim != 3
        or not segments
        or red_mask is None
        or green_mask.shape != red_mask.shape
        or not cv2.countNonZero(red_mask)
    ):
        return segments, []
    green = green_mask > 0
    red = red_mask > 0
    height, width = green.shape[:2]
    stroke = max(1.0, float(stroke_width))
    lateral = max(1, min(3, int(round(stroke))))
    side_window = max(2, min(8, int(round(3.0 * stroke))))
    records: list[dict[str, Any]] = [
        record for record in terminal_records if isinstance(record, dict)
    ]
    if not records:
        return segments, []

    def support(mask: np.ndarray, x: int, y: int) -> bool:
        x0, x1 = max(0, x - lateral), min(width, x + lateral + 1)
        y0, y1 = max(0, y - lateral), min(height, y + lateral + 1)
        return bool(np.any(mask[y0:y1, x0:x1]))

    def point_of(record: dict[str, Any]) -> tuple[float, float] | None:
        point = record.get("point_top_left") or record.get("skeleton_point_top_left")
        if not isinstance(point, dict):
            return None
        try:
            return float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            return None

    def record_accepts(
        record: dict[str, Any],
        orientation: str,
        coordinate: float,
        transition_axis: float,
    ) -> bool:
        point = point_of(record)
        if point is None:
            return False
        px, py = point
        record_axis = py if orientation == "horizontal" else px
        record_along = px if orientation == "horizontal" else py
        if abs(record_axis - coordinate) > max(3.0, 2.5 * stroke):
            return False
        side = str(record.get("side") or "").lower()
        outward = {
            "left": -1.0,
            "right": 1.0,
            "top": -1.0,
            "bottom": 1.0,
        }.get(side)
        if orientation == "horizontal" and side not in {"left", "right"}:
            return False
        if orientation == "vertical" and side not in {"top", "bottom"}:
            return False
        if outward is not None and (transition_axis - record_along) * outward < -stroke:
            return False
        scale = max(1.0, float(record.get("component_scale") or 32.0))
        max_distance = min(32.0, max(6.0, 0.9 * scale))
        return abs(transition_axis - record_along) <= max_distance

    transition_points: list[tuple[float, float]] = []
    for segment in segments:
        orientation = str(segment.get("orientation") or "")
        if orientation not in {"horizontal", "vertical"}:
            continue
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        if orientation == "horizontal":
            coordinate = (first[1] + second[1]) / 2.0
            low, high = sorted((first[0], second[0]))
        else:
            coordinate = (first[0] + second[0]) / 2.0
            low, high = sorted((first[1], second[1]))
        if high - low < max(5.0, 4.0 * stroke):
            continue
        labels: list[str] = []
        # Include a small raster context outside the current Hough child.  A
        # previous terminal split may have placed the beginning of a red run
        # exactly at this child's endpoint; without context the green run on
        # the other side is invisible and the late split is missed.
        axis_values = list(
            range(
                int(math.ceil(low)) - side_window,
                int(math.floor(high)) + side_window + 1,
            )
        )
        for axis in axis_values:
            x = int(round(axis)) if orientation == "horizontal" else int(round(coordinate))
            y = int(round(coordinate)) if orientation == "horizontal" else int(round(axis))
            green_hit = support(green, x, y)
            red_hit = support(red, x, y)
            labels.append("R" if red_hit else "G" if green_hit else ".")
        index = 0
        while index < len(labels):
            if labels[index] != "R":
                index += 1
                continue
            run_start = index
            while index + 1 < len(labels) and labels[index + 1] == "R":
                index += 1
            run_end = index
            before_start = max(0, run_start - side_window)
            after_end = min(len(labels), run_end + side_window + 1)
            before_green = any(label == "G" for label in labels[before_start:run_start])
            after_green = any(label == "G" for label in labels[run_end + 1:after_end])
            run_length = run_end - run_start + 1
            # A long red interval is a symbol body/text stroke, not a
            # terminal transition.  The adaptive upper bound scales with the
            # nearby component record below.
            if before_green and after_green and run_length <= max(4, int(round(4.0 * max(1.0, stroke)))):
                transition_axis = float(axis_values[run_start] + axis_values[run_end]) / 2.0
                if transition_axis < low - 0.5 or transition_axis > high + 0.5:
                    index += 1
                    continue
                if any(
                    record_accepts(record, orientation, coordinate, transition_axis)
                    for record in records
                ):
                    transition_points.append(
                        (transition_axis, coordinate)
                        if orientation == "horizontal"
                        else (coordinate, transition_axis)
                    )
            index += 1

    if not transition_points:
        return segments, []
    unique_points: list[tuple[float, float]] = []
    for point in transition_points:
        if not any(math.hypot(point[0] - old[0], point[1] - old[1]) <= max(1.0, stroke) for old in unique_points):
            unique_points.append(point)
    split = _split_axis_segments_at_points(
        segments,
        unique_points,
        tolerance=max(2.0, min(6.0, 2.5 * stroke)),
    )
    for item in split:
        first, second = _segment_points(item)
        if any(
            item.get("orientation") == ("horizontal" if abs(point[1] - first[1]) <= max(2.0, 2.5 * stroke) else "vertical")
            and (
                min(first[0], second[0]) - 1.0 <= point[0] <= max(first[0], second[0]) + 1.0
                if item.get("orientation") == "horizontal"
                else min(first[1], second[1]) - 1.0 <= point[1] <= max(first[1], second[1]) + 1.0
            )
            for point in unique_points
        ):
            item["local_color_transition_split"] = True
    return split, unique_points


def _split_at_observed_nodes(
    segments: list[dict[str, Any]],
    terminal_records: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    *,
    stroke_width: float,
    preserve_parent: bool = False,
) -> list[dict[str, Any]]:
    """Split existing wire candidates at image-supported graph nodes.

    Hough merging is useful for recovering a long antialiased run, but it can
    hide the real graph structure when a pin or branch lies in the middle of
    that run.  This function never creates a path: it only partitions an
    existing, skeleton-validated segment at a terminal or junction whose
    observed point lies on that segment.  Crossing candidates are included as
    geometric split points; electrical joining remains a later topology
    decision.
    """
    if not segments:
        return []
    tolerance = max(1.5, min(5.0, 1.5 * max(1.0, float(stroke_width))))
    node_points: list[tuple[float, float, str, dict[str, Any] | None]] = []
    for record in terminal_records:
        # Global Hough is raster-supported and therefore lies on the observed
        # skeleton node.  For a validated pin, however, the protocol edge is
        # bounded by the measured physical terminal/boundary point, not by
        # the one-pixel skeleton attachment that may sit inside the red
        # terminal.  Generic component-boundary observations still use their
        # raster intersection because they do not have a validated pin tip.
        if record.get("pin_anchor_evidence", False):
            point = record.get("point_top_left") or record.get("skeleton_point_top_left")
        else:
            point = record.get("skeleton_point_top_left") or record.get("point_top_left")
        if isinstance(point, dict):
            try:
                node_points.append((float(point["x"]), float(point["y"]), "terminal", record))
            except (KeyError, TypeError, ValueError):
                pass
    for record in junctions:
        point = record.get("point_top_left")
        if isinstance(point, dict):
            try:
                node_points.append((float(point["x"]), float(point["y"]), "junction", record))
            except (KeyError, TypeError, ValueError):
                pass

    output: list[dict[str, Any]] = []
    for segment in segments:
        p1, p2 = _segment_points(segment)
        horizontal = abs(p2[0] - p1[0]) >= abs(p2[1] - p1[1])
        if horizontal:
            coordinate = (p1[1] + p2[1]) / 2.0
            start, end = sorted((p1[0], p2[0]))
            points: list[tuple[float, float]] = [(start, coordinate), (end, coordinate)]
            for x, y, _kind, _record in node_points:
                if start + tolerance < x < end - tolerance and abs(y - coordinate) <= tolerance:
                    points.append((x, coordinate))
            points.sort(key=lambda value: value[0])
        else:
            coordinate = (p1[0] + p2[0]) / 2.0
            start, end = sorted((p1[1], p2[1]))
            points = [(coordinate, start), (coordinate, end)]
            for x, y, _kind, _record in node_points:
                if start + tolerance < y < end - tolerance and abs(x - coordinate) <= tolerance:
                    points.append((coordinate, y))
            points.sort(key=lambda value: value[1])

        unique: list[tuple[float, float]] = []
        for point in points:
            if not unique or math.hypot(point[0] - unique[-1][0], point[1] - unique[-1][1]) > 1.0:
                unique.append(point)
        was_split = len(unique) > 2
        for left, right in zip(unique, unique[1:]):
            if math.hypot(right[0] - left[0], right[1] - left[1]) <= 0.5:
                continue
            item = dict(segment)
            item["points_top_left"] = [_as_point(left), _as_point(right)]
            item["split_from_observed_node"] = len(unique) > 2
            raw_endpoint_types = item.get("endpoint_types", [])
            raw_terminal_bindings = item.get("terminal_bindings", [])
            endpoint_types = list(raw_endpoint_types) if isinstance(raw_endpoint_types, list) else []
            terminal_bindings = list(raw_terminal_bindings) if isinstance(raw_terminal_bindings, list) else []
            while len(endpoint_types) < 2:
                endpoint_types.append("wire_node")
            while len(terminal_bindings) < 2:
                terminal_bindings.append(None)
            for endpoint_index, endpoint in enumerate((left, right)):
                nearest_node = None
                nearest_distance = float("inf")
                for node_x, node_y, node_kind, node_record in node_points:
                    distance = math.hypot(endpoint[0] - node_x, endpoint[1] - node_y)
                    if distance <= tolerance and distance < nearest_distance:
                        nearest_distance = distance
                        nearest_node = (node_kind, node_record)
                if nearest_node is None:
                    continue
                node_kind, node_record = nearest_node
                if node_kind == "terminal":
                    endpoint_types[endpoint_index] = "component_terminal"
                    if isinstance(node_record, dict):
                        terminal_bindings[endpoint_index] = {
                            key: node_record[key]
                            for key in ("component_key", "pin_key", "component_index", "side")
                            if node_record.get(key) is not None
                        }
                elif node_kind == "junction":
                    endpoint_types[endpoint_index] = "wire_junction"
            item["endpoint_types"] = endpoint_types
            item["terminal_bindings"] = terminal_bindings
            output.append(item)
        if preserve_parent and was_split:
            parent = dict(segment)
            parent["parent_preserved_for_observed_split"] = True
            output.append(parent)
    return output


def _split_axis_segments_at_points(
    segments: list[dict[str, Any]],
    points: Iterable[tuple[float, float]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Split image-supported axis runs at nearby validated terminal points.

    Hough sees a long physical run as one segment even when the contest JSON
    splits it at a pin terminal.  Splitting is allowed only for a validated
    local pin anchor and only when the point is close to the observed run;
    arbitrary bbox sides and public edge counts never trigger this function.
    """
    split_points = [(float(x), float(y)) for x, y in points]
    if not segments or not split_points:
        return segments
    result: list[dict[str, Any]] = []
    for segment in segments:
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        x1, y1 = first
        x2, y2 = second
        horizontal = abs(y2 - y1) <= max(1.5, tolerance * 0.6)
        vertical = abs(x2 - x1) <= max(1.5, tolerance * 0.6)
        if not (horizontal or vertical):
            result.append(segment)
            continue
        if horizontal:
            coordinate = (y1 + y2) / 2.0
            low, high = sorted((x1, x2))
            candidates = [
                (px, coordinate)
                for px, py in split_points
                if low + 0.5 < px < high - 0.5 and abs(py - coordinate) <= tolerance
            ]
            cuts = sorted({low, *(item[0] for item in candidates), high})
            reverse = x2 < x1
            pairs = list(zip(cuts, cuts[1:]))
            if reverse:
                pairs = [(right, left) for left, right in reversed(pairs)]
            for left, right in pairs:
                if abs(right - left) < 0.75:
                    continue
                item = dict(segment)
                item["points_top_left"] = [_as_point((left, coordinate)), _as_point((right, coordinate))]
                item["source"] = "wire_hough_split_at_pin"
                item["pin_split_evidence"] = True
                result.append(item)
        else:
            coordinate = (x1 + x2) / 2.0
            low, high = sorted((y1, y2))
            candidates = [
                (coordinate, py)
                for px, py in split_points
                if low + 0.5 < py < high - 0.5 and abs(px - coordinate) <= tolerance
            ]
            cuts = sorted({low, *(item[1] for item in candidates), high})
            reverse = y2 < y1
            pairs = list(zip(cuts, cuts[1:]))
            if reverse:
                pairs = [(right, left) for left, right in reversed(pairs)]
            for left, right in pairs:
                if abs(right - left) < 0.75:
                    continue
                item = dict(segment)
                item["points_top_left"] = [_as_point((coordinate, left)), _as_point((coordinate, right))]
                item["source"] = "wire_hough_split_at_pin"
                item["pin_split_evidence"] = True
                result.append(item)
    return result


def _observed_endpoint_skeleton_candidates(
    segments: list[dict[str, Any]],
    terminal_records: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Keep skeleton paths that terminate at an observed graph node.

    This is a bounded recall supplement: a skeleton fragment is eligible
    only when at least one endpoint is close to a validated pin terminal or a
    classified junction.  Free text-like skeleton fragments therefore do not
    enter the output merely because they are dark/green pixels.
    """
    evidence: list[tuple[float, float]] = []
    for record in [*terminal_records, *junctions]:
        point = record.get("point_top_left")
        if isinstance(point, dict):
            try:
                evidence.append((float(point["x"]), float(point["y"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not evidence:
        return []
    radius = max(2.0, min(7.0, 2.5 * max(1.0, float(stroke_width))))
    output: list[dict[str, Any]] = []
    for segment in segments:
        p1, p2 = _segment_points(segment)
        if any(
            min(math.hypot(p1[0] - x, p1[1] - y), math.hypot(p2[0] - x, p2[1] - y)) <= radius
            for x, y in evidence
        ):
            item = dict(segment)
            item["source"] = "wire_observed_node_skeleton_path"
            item["observed_endpoint_evidence"] = True
            output.append(item)
    return output


def _evidence_connected_skeleton_candidates(
    skeleton: np.ndarray,
    segments: list[dict[str, Any]],
    terminal_records: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Keep every graph interval in a skeleton component with local evidence.

    Endpoint-only recovery misses the middle intervals of a pin--junction
    network.  This helper labels the 4-connected skeleton and admits a path
    interval only when its raster component contains a validated terminal or
    a geometrically classified junction.  It does not infer connectivity from
    a bbox, pin-library order, edge counts, or colour alone.
    """
    if not segments or skeleton.size == 0:
        return []
    evidence: list[tuple[float, float]] = []
    for record in [*terminal_records, *junctions]:
        point = record.get("point_top_left")
        if isinstance(point, dict):
            try:
                evidence.append((float(point["x"]), float(point["y"])))
            except (KeyError, TypeError, ValueError):
                continue
    if not evidence:
        return []
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=4
    )
    if count <= 1:
        return []
    radius = max(2, int(round(2.0 * max(1.0, float(stroke_width)))))
    evidence_labels: set[int] = set()
    evidence_label_counts: dict[int, int] = {}
    height, width = labels.shape[:2]
    for px, py in evidence:
        cx, cy = int(round(px)), int(round(py))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        local = labels[y0:y1, x0:x1]
        local_labels = [int(value) for value in np.unique(local) if int(value) > 0]
        evidence_labels.update(local_labels)
        for label in local_labels:
            evidence_label_counts[label] = evidence_label_counts.get(label, 0) + 1
    if not evidence_labels:
        return []

    def label_near(point: tuple[float, float]) -> int:
        cx, cy = int(round(point[0])), int(round(point[1]))
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        local = labels[y0:y1, x0:x1]
        values = [int(value) for value in np.unique(local) if int(value) > 0]
        return values[0] if values else 0

    output: list[dict[str, Any]] = []
    for segment in segments:
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        label = label_near(first)
        if label <= 0:
            label = label_near(second)
        if label not in evidence_labels:
            continue
        item = dict(segment)
        item["source"] = "wire_skeleton_evidence_connected_graph"
        item["skeleton_component_evidence"] = True
        item["skeleton_component_label"] = label
        item["skeleton_component_area"] = int(stats[label, cv2.CC_STAT_AREA])
        item["skeleton_component_evidence_count"] = int(evidence_label_counts.get(label, 1))
        output.append(item)
    return output


def _merge_split_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate Hough fragments without bridging real symbol gaps."""
    if not segments:
        return []
    ordered = sorted(
        segments,
        key=lambda item: (
            item["orientation"],
            round(
                (item["points_top_left"][0]["y"] + item["points_top_left"][1]["y"])
                / 2.0
                if item["orientation"] == "horizontal"
                else (item["points_top_left"][0]["x"] + item["points_top_left"][1]["x"])
                / 2.0,
            1),
            item["points_top_left"][0]["x"]
            if item["orientation"] == "horizontal"
            else item["points_top_left"][0]["y"],
        ),
    )
    merged: list[dict[str, Any]] = []
    for item in ordered:
        p1, p2 = _segment_points(item)
        horizontal = item["orientation"] == "horizontal"
        coordinate = (p1[1] + p2[1]) / 2.0 if horizontal else (p1[0] + p2[0]) / 2.0
        start, end = sorted((p1[0], p2[0]) if horizontal else (p1[1], p2[1]))
        previous = merged[-1] if merged else None
        if previous is not None:
            q1, q2 = _segment_points(previous)
            previous_coordinate = (
                (q1[1] + q2[1]) / 2.0
                if horizontal
                else (q1[0] + q2[0]) / 2.0
            )
            previous_start, previous_end = sorted(
                (q1[0], q2[0]) if horizontal else (q1[1], q2[1])
            )
            if (
                previous["orientation"] == item["orientation"]
                and abs(previous_coordinate - coordinate) <= 3.0
                and start <= previous_end + 5.0
                and end >= previous_start - 5.0
            ):
                start = min(start, previous_start)
                end = max(end, previous_end)
                coordinate = (previous_coordinate + coordinate) / 2.0
                if horizontal:
                    previous["points_top_left"] = [
                        {"x": round(start, 2), "y": round(coordinate, 2)},
                        {"x": round(end, 2), "y": round(coordinate, 2)},
                    ]
                else:
                    previous["points_top_left"] = [
                        {"x": round(coordinate, 2), "y": round(start, 2)},
                        {"x": round(coordinate, 2), "y": round(end, 2)},
                    ]
                previous["confidence"] = max(previous["confidence"], item["confidence"])
                continue
        if end - start < 5.0:
            continue
        if horizontal:
            points = [
                {"x": round(start, 2), "y": round(coordinate, 2)},
                {"x": round(end, 2), "y": round(coordinate, 2)},
            ]
        else:
            points = [
                {"x": round(coordinate, 2), "y": round(start, 2)},
                {"x": round(coordinate, 2), "y": round(end, 2)},
            ]
        merged.append(
            {
                "points_top_left": points,
                "orientation": item["orientation"],
                "source": item["source"],
                "confidence": item["confidence"],
            }
        )
    for index, segment in enumerate(merged, start=1):
        segment["id"] = f"wire_{index}"
    return merged


def _wire_ink_mask(
    image: np.ndarray,
    *,
    include_red: bool = False,
    component_boxes: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Select the schematic wire layer while excluding symbol/pin ink.

    In the supplied rasters, green is the physical wire layer and red is
    primarily the component/pin drawing layer.  Red pixels are therefore not
    treated as wires by default; direct red-pin connections are reconstructed
    later from terminal/pin geometry.  ``include_red`` is retained as an
    explicit opt-in for a monochrome/source variant whose documentation proves
    that red is a wire layer.
    """
    if image.ndim != 3:
        return _black_line_mask(image, component_boxes=component_boxes)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    # Use a strict core to decide whether this is a green-wire rendering, then
    # expand around that core to recover anti-aliased edges.  A few cyan/blue
    # pixels in an otherwise black schematic must not switch the whole image
    # into the green branch.
    core_green = (hue >= 35) & (hue <= 95) & (saturation >= 50) & (value >= 45)
    core = np.where(core_green, 255, 0).astype(np.uint8)
    core_count = int(cv2.countNonZero(core))
    if core_count < 80:
        return _black_line_mask(image, component_boxes=component_boxes)
    coloured = (saturation >= 20) & (value >= 30)
    green = (hue >= 25) & (hue <= 105)
    # Only relaxed green pixels adjacent to a real core stroke are retained.
    # This keeps anti-aliased boundaries without admitting unrelated coloured
    # annotations elsewhere in the page.
    expanded_core = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    red = (hue <= 10) | (hue >= 170)
    wire_layer = coloured & ((green & expanded_core) | (include_red & red))
    if int(wire_layer.sum()) >= 80:
        # Keep the colour layer intact.  Short terminal stubs, one-pixel
        # junction arms, and anti-aliased bends are real wire evidence even
        # when they do not survive a global morphology opening.  The global
        # Hough stage still applies its measured-stroke opening; the
        # skeleton/local stages need the unabridged raster so that they can
        # recover these short paths from a validated terminal.  Green
        # annotations are rejected later by component/junction connectivity,
        # not by deleting pixels before the local evidence stage.
        return np.where(wire_layer, 255, 0).astype(np.uint8)
    # The other permitted source mode is a monochrome schematic.  Do not
    # skeletonize every dark pixel: text and component labels are dark too.
    # The line-prior mask keeps long horizontal/vertical strokes and lets the
    # pin-local recovery use the unfiltered achromatic mask when a short
    # terminal is not long enough for the global prior.
    return _black_line_mask(image, component_boxes=component_boxes)


def _axis_supported_graph_skeleton(
    mask: np.ndarray,
    *,
    stroke_width: float,
) -> np.ndarray:
    """Build a graph-only skeleton without deleting the source wire mask.

    Green renderers also use green for no-connect crosses and small
    annotations.  Those marks should remain available to local terminal
    inspection, but they should not become global graph branches.  The
    support kernel is derived from the measured stroke width and only affects
    this graph view; Hough and local ROI extraction continue to see the raw
    mask.
    """
    raw = np.where(mask > 0, 255, 0).astype(np.uint8)
    if cv2.countNonZero(raw) == 0:
        return raw
    support_length = max(3, min(11, int(round(2.0 * max(1.0, float(stroke_width)) + 1.0))))
    horizontal = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (support_length, 1)),
    )
    vertical = cv2.morphologyEx(
        raw,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, support_length)),
    )
    support = cv2.bitwise_or(horizontal, vertical)
    radius = max(1, min(3, int(round(max(1.0, float(stroke_width))))))
    support = cv2.dilate(
        support,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    )
    return _thin(cv2.bitwise_and(raw, support))


def _black_line_mask(
    image: np.ndarray,
    *,
    component_boxes: list[tuple[float, float, float, float]] | None = None,
) -> np.ndarray:
    """Build a conservative wire mask for the all-black source mode.

    This is intentionally renderer-agnostic.  It uses the measured stroke
    width to find horizontal/vertical line support, instead of using a case
    number, a target edge count, or a fixed image-size threshold.  Short
    terminal stubs are recovered later in the pin-local path search from the
    achromatic base mask.
    """
    if image.ndim == 2:
        gray = image
        saturation = np.zeros_like(gray)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Include anti-aliased gray pixels, but retain the low-saturation guard so
    # coloured labels are not treated as black wire.
    base = np.where((gray < 225) & (saturation <= 75), 255, 0).astype(np.uint8)
    if cv2.countNonZero(base) < 80:
        return base

    stroke = _estimate_stroke_width(base)
    # A line must survive a short opening in at least one orthogonal
    # direction.  The length is tied to measured raster stroke width, not to
    # the resolution of one particular public case.
    # A long opening is a common source of missed short wires.  The line
    # prior is intentionally short; terminal-connected fragments are restored
    # separately below.
    # Keep the global prior conservative enough to suppress ordinary glyph
    # strokes.  Short pin stubs are restored in the component-local corridor
    # below, where there is geometric terminal evidence.
    support_length = max(3, min(13, int(round(4.0 * stroke + 1.0))))
    horizontal = cv2.morphologyEx(
        base,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (support_length, 1)),
    )
    vertical = cv2.morphologyEx(
        base,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, support_length)),
    )
    seed = cv2.bitwise_or(horizontal, vertical)
    if cv2.countNonZero(seed) < 20:
        return base

    # Recover the original stroke around a line seed, but do not recover all
    # dark text pixels.  This preserves junction thickness while keeping most
    # glyphs out of the global skeleton.
    radius = max(1, min(3, int(round(stroke))))
    support = cv2.dilate(
        seed,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
    )
    result = cv2.bitwise_and(base, support)
    # A page border is a drawing frame, not a schematic wire.  This removes
    # only rows/columns with near-global support and is independent of case
    # dimensions or target edge counts.
    _mask_page_frame(result)
    result = _restore_external_terminal_strokes(
        result,
        base,
        component_boxes or [],
        stroke_width=stroke,
    )
    return result


def _restore_external_terminal_strokes(
    mask: np.ndarray,
    source: np.ndarray,
    component_boxes: list[tuple[float, float, float, float]],
    *,
    stroke_width: float,
) -> np.ndarray:
    """Restore short achromatic wire evidence outside detected components.

    The global black-line prior intentionally rejects compact glyphs.  That
    prior can also reject a real short pin-to-wire stub, so restore only a
    scale-derived band outside each component body.  This does not add the
    component interior and does not connect unrelated regions globally.
    """
    if not component_boxes or not cv2.countNonZero(source):
        return mask
    height, width = mask.shape[:2]
    output = mask.copy()
    # The expression above is intentionally bounded by stroke width; use a
    # minimum corridor for small rendered symbols and cap it for large ones.
    for x1, y1, x2, y2 in component_boxes:
        span = max(1.0, min(x2 - x1, y2 - y1))
        local_reach = max(5, min(64, int(round(0.18 * span))))
        lateral = max(2, min(6, int(round(1.5 * stroke_width))))
        corridor = np.zeros_like(source)
        left = max(0, int(round(x1 - local_reach)))
        right = min(width - 1, int(round(x2 + local_reach)))
        top = max(0, int(round(y1 - local_reach)))
        bottom = min(height - 1, int(round(y2 + local_reach)))
        # External bands only.  A small overlap at the body boundary keeps a
        # true terminal transition intact; the body itself is never restored.
        corridor[max(0, int(round(y1 - lateral))):min(height, int(round(y2 + lateral + 1))), left:min(width, int(round(x1 + lateral + 1)))] = 255
        corridor[max(0, int(round(y1 - lateral))):min(height, int(round(y2 + lateral + 1))), max(0, int(round(x2 - lateral))):right + 1] = 255
        corridor[top:min(height, int(round(y1 + lateral + 1))), max(0, int(round(x1 - lateral))):min(width, int(round(x2 + lateral + 1)))] = 255
        corridor[max(0, int(round(y2 - lateral))):bottom + 1, max(0, int(round(x1 - lateral))):min(width, int(round(x2 + lateral + 1)))] = 255
        local_source = cv2.bitwise_and(source, corridor)
        # Require a tiny directional run, not a single isolated dark pixel.
        run = max(3, min(9, int(round(2.0 * stroke_width + 1.0))))
        horizontal = cv2.morphologyEx(
            local_source,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (run, 1)),
        )
        vertical = cv2.morphologyEx(
            local_source,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, run)),
        )
        output = cv2.bitwise_or(output, cv2.bitwise_or(horizontal, vertical))
    return output


def _achromatic_wire_source(image: np.ndarray) -> np.ndarray:
    """Return the unfiltered black-line source used only for local recovery."""
    if image.ndim == 2:
        return np.where(image < 210, 255, 0).astype(np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.where((gray < 225) & (hsv[:, :, 1] <= 75), 255, 0).astype(np.uint8)


def _red_terminal_mask(image: np.ndarray) -> np.ndarray:
    """Return red pixels for local pin-terminal transition checks only."""
    if image.ndim != 3:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    red = ((hue <= 10) | (hue >= 170)) & (saturation >= 45) & (value >= 45)
    return np.where(red, 255, 0).astype(np.uint8)


def _has_colored_wire_layer(image: np.ndarray) -> bool:
    if image.ndim != 3:
        return False
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = (hue >= 35) & (hue <= 95) & (saturation >= 50) & (value >= 45)
    core = np.where(green, 255, 0).astype(np.uint8)
    if cv2.countNonZero(core) < 80:
        return False
    support = cv2.bitwise_or(
        cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))),
        cv2.morphologyEx(core, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))),
    )
    return cv2.countNonZero(support) >= 80


def _remove_tiny_ink_components(mask: np.ndarray, *, min_area: int) -> np.ndarray:
    """Drop isolated anti-alias specks without cutting connected wire paths.

    This is intentionally applied only to high-resolution rasters and only to
    components that are disconnected from every other ink pixel.  A short
    wire connected to a longer path is therefore preserved, even when its
    own geometric edge is shorter than the threshold.
    """
    if min_area <= 1 or not cv2.countNonZero(mask):
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            keep[labels == label] = 255
    return keep


def _estimate_stroke_width(mask: np.ndarray) -> float:
    """Estimate the raster stroke width used for all morphology parameters."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not cv2.countNonZero(binary):
        return 1.0
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    values = distance[distance > 0]
    if values.size == 0:
        return 1.0
    # The lower-middle quantile is less affected by large filled symbols and
    # remains stable between JLC-sized and high-resolution Altium rasters.
    width = 2.0 * float(np.percentile(values, 40.0))
    return max(1.0, min(32.0, width))


def _odd_kernel_size(value: float, *, minimum: int = 1, maximum: int = 9) -> int:
    # Keep this helper deliberately small: large closing kernels bridge
    # adjacent pins and violate the real-wire requirement.
    size = max(minimum, min(maximum, int(round(value))))
    return size if size % 2 else size + 1 if size < maximum else size - 1


def _thin(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # OpenCV contrib is included in the contest CPU environment.  The small
    # fallback preserves correctness if a minimal OpenCV build is used.
    skeleton = np.zeros_like(binary)
    work = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return skeleton


def _component_terminal_nodes(
    skeleton: np.ndarray,
    component_boxes: list[tuple[float, float, float, float]],
    *,
    tolerance: int = 10,
) -> set[tuple[int, int]]:
    """Return one observed skeleton point per terminal-side intersection."""
    height, width = skeleton.shape[:2]
    points: set[tuple[int, int]] = set()

    for x1, y1, x2, y2 in component_boxes:
        # The detector bbox is usually the body bbox, while the contest pin
        # tip can be several pixels outside it.  Scale the search by the body
        # size instead of using one image-wide fixed expansion.
        span = max(1.0, min(x2 - x1, y2 - y1))
        local_tolerance = max(
            int(tolerance),
            min(128, int(round(0.18 * max(1.0, min(x2 - x1, y2 - y1))))),
        )
        edge_margin = max(2, local_tolerance // 3)
        sides = (
            ("left", int(round(x1)), int(round(y1)), int(round(y2))),
            ("right", int(round(x2)), int(round(y1)), int(round(y2))),
            ("top", int(round(y1)), int(round(x1)), int(round(x2))),
            ("bottom", int(round(y2)), int(round(x1)), int(round(x2))),
        )
        for side, coordinate, start, end in sides:
            observed: list[tuple[int, int]] = []
            if side in {"left", "right"}:
                for y in range(max(0, start + edge_margin), min(height, end - edge_margin)):
                    if side == "left":
                        band_start = max(0, coordinate - local_tolerance)
                        band_end = max(band_start, coordinate - edge_margin)
                    else:
                        band_start = min(width, coordinate + edge_margin)
                        band_end = min(width, coordinate + local_tolerance + 1)
                    xs = np.flatnonzero(skeleton[y, band_start:band_end])
                    if xs.size:
                        x = band_start + int(xs[np.argmin(np.abs(xs + band_start - coordinate))])
                        observed.append((x, y))
            else:
                for x in range(max(0, start + edge_margin), min(width, end - edge_margin)):
                    if side == "top":
                        band_start = max(0, coordinate - local_tolerance)
                        band_end = max(band_start, coordinate - edge_margin)
                    else:
                        band_start = min(height, coordinate + edge_margin)
                        band_end = min(height, coordinate + local_tolerance + 1)
                    ys = np.flatnonzero(skeleton[band_start:band_end, x])
                    if ys.size:
                        y = band_start + int(ys[np.argmin(np.abs(ys + band_start - coordinate))])
                        observed.append((x, y))
            # A thick terminal creates several adjacent skeleton pixels. Keep
            # its median point; separate terminals remain separated by pitch.
            groups: list[list[tuple[int, int]]] = []
            for point in observed:
                axis = point[1] if side in {"left", "right"} else point[0]
                if not groups:
                    groups.append([point])
                    continue
                previous = groups[-1][-1]
                previous_axis = previous[1] if side in {"left", "right"} else previous[0]
                if axis - previous_axis <= 4:
                    groups[-1].append(point)
                else:
                    groups.append([point])
            for group in groups:
                group.sort(key=lambda point: abs((point[0] if side in {"left", "right"} else point[1]) - coordinate))
                points.add(group[0])
    return points


def _project_to_component_boundary(
    point: tuple[int, int],
    component_boxes: list[tuple[float, float, float, float]],
) -> tuple[dict[str, Any] | None, float]:
    """Project an observed external skeleton point to the nearest bbox side.

    The raster skeleton point is only an observation.  The contest pin/wire
    point is the actual component-boundary intersection, so the exported
    terminal is snapped to that side rather than left at an arbitrary pixel
    offset.
    """
    x, y = map(float, point)
    best: tuple[float, dict[str, Any]] | None = None
    for index, (x1, y1, x2, y2) in enumerate(component_boxes):
        margin = max(7.0, min(128.0, 0.18 * max(1.0, min(x2 - x1, y2 - y1))))
        near_vertical_span = y1 - margin <= y <= y2 + margin
        near_horizontal_span = x1 - margin <= x <= x2 + margin
        candidates = [
            (abs(x - x1), {"component_index": index, "side": "left", "point": (x1, min(max(y, y1), y2))})
            if near_vertical_span else None,
            (abs(x - x2), {"component_index": index, "side": "right", "point": (x2, min(max(y, y1), y2))})
            if near_vertical_span else None,
            (abs(y - y1), {"component_index": index, "side": "top", "point": (min(max(x, x1), x2), y1)})
            if near_horizontal_span else None,
            (abs(y - y2), {"component_index": index, "side": "bottom", "point": (min(max(x, x1), x2), y2)})
            if near_horizontal_span else None,
        ]
        for pair in candidates:
            if pair is None:
                continue
            distance, candidate = pair
            if best is None or distance < best[0]:
                best = (distance, candidate)
    if best is None:
        return None, float("inf")
    candidate = dict(best[1])
    candidate["point"] = _as_point(candidate["point"])
    return candidate, float(best[0])


def _terminal_records(
    terminal_nodes: set[tuple[int, int]],
    component_boxes: list[tuple[float, float, float, float]],
    *,
    max_distance: float = 128.0,
    pin_anchor_by_node: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    used: set[tuple[int, int, str]] = set()
    pin_anchor_by_node = pin_anchor_by_node or {}
    pin_anchor_nodes = list(pin_anchor_by_node)
    for node in sorted(terminal_nodes, key=lambda value: (value[1], value[0])):
        anchor = pin_anchor_by_node.get(node)
        if anchor is None:
            # Prefer a pin-anchored record over the generic bbox record when
            # both observations refer to the same physical terminal.
            if any(math.hypot(node[0] - other[0], node[1] - other[1]) <= 3.0 for other in pin_anchor_nodes):
                continue
        if anchor is not None:
            point = anchor["point_top_left"]
            side = str(anchor.get("side") or "unknown")
            key = (round(float(point["x"])), round(float(point["y"])), side)
            if key in used:
                continue
            used.add(key)
            records.append(
                {
                    "point_top_left": _as_point((float(point["x"]), float(point["y"]))),
                    "component_boundary_reference_top_left": _as_point(
                        (float(point["x"]), float(point["y"]))
                    ),
                    "skeleton_point_top_left": _as_point((float(node[0]), float(node[1]))),
                    "component_index": anchor.get("component_index"),
                    "component_key": anchor.get("component_key"),
                    "pin_key": anchor.get("pin_key"),
                    "side": side,
                    "boundary_distance_px": round(
                        math.hypot(
                            float(point["x"]) - float(node[0]),
                            float(point["y"]) - float(node[1]),
                        ),
                        3,
                    ),
                    "source": "pin_anchor_skeleton_connection",
                    "pin_anchor_evidence": True,
                    "pin_path_evidence": bool(anchor.get("pin_path_evidence", False)),
                    "skeleton_distance_px": anchor.get("skeleton_distance_px"),
                    "terminal_transition": anchor.get("terminal_transition"),
                }
            )
            continue
        candidate, distance = _project_to_component_boundary(node, component_boxes)
        if candidate is None or distance > max_distance:
            continue
        # Keep the observed skeleton intersection as the actual terminal.
        # The projection to the body bbox is only a reference because the
        # public component bbox may exclude the external pin stem.
        observed_point = _as_point((float(node[0]), float(node[1])))
        point = observed_point
        key = (round(float(point["x"])), round(float(point["y"])), str(candidate["side"]))
        if key in used:
            continue
        used.add(key)
        records.append(
            {
                "point_top_left": point,
                "component_boundary_reference_top_left": candidate["point"],
                "skeleton_point_top_left": _as_point((float(node[0]), float(node[1]))),
                "component_index": int(candidate["component_index"]),
                "side": candidate["side"],
                "boundary_distance_px": round(distance, 3),
                "source": "component_boundary_skeleton_intersection",
            }
        )
    return records


def _component_red_green_terminal_seeds(
    image: np.ndarray,
    component_boxes: list[tuple[float, float, float, float]],
    *,
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    skeleton_mask: np.ndarray,
    global_segments: list[dict[str, Any]] | None,
    stroke_width: float,
    existing_records: Iterable[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Find non-IC terminal seeds from a local red-to-green transition.

    Non-IC symbols do not have formal pin anchors.  Their visible terminal is
    nevertheless measurable: a red symbol/terminal stroke lies immediately
    beside the component body and the green wire continues outward along the
    side normal.  This helper records only that local geometric fact.  It
    does not add red pixels to the global mask and never assigns a pin number.
    """
    if image.ndim != 3 or red_mask is None:
        return []
    if green_mask.shape != red_mask.shape or green_mask.shape != skeleton_mask.shape:
        return []
    if not cv2.countNonZero(red_mask) or not cv2.countNonZero(green_mask):
        return []
    height, width = green_mask.shape[:2]
    green = green_mask > 0
    red = red_mask > 0
    stroke = max(1.0, float(stroke_width))
    directions = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    seeds: list[dict[str, Any]] = []
    existing_axis: dict[tuple[Any, str], list[float]] = {}
    for record in existing_records:
        if not isinstance(record, dict):
            continue
        point = record.get("point_top_left") or record.get("skeleton_point_top_left")
        side = str(record.get("side") or "").lower()
        index = record.get("component_index")
        if not isinstance(point, dict) or side not in directions or index is None:
            continue
        try:
            axis = float(point["y"] if side in {"left", "right"} else point["x"])
        except (KeyError, TypeError, ValueError):
            continue
        existing_axis.setdefault((index, side), []).append(axis)

    def global_support(x: int, y: int, side: str, radius: int) -> bool:
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        if np.any(skeleton_mask[y0:y1, x0:x1]):
            return True
        expected = "horizontal" if side in {"left", "right"} else "vertical"
        for segment in global_segments or []:
            if segment.get("orientation") != expected:
                continue
            try:
                first, second = _segment_points(segment)
            except (KeyError, TypeError, ValueError):
                continue
            if expected == "horizontal":
                coordinate = (first[1] + second[1]) / 2.0
                low, high = sorted((first[0], second[0]))
                if abs(coordinate - y) <= max(2.0, 2.0 * stroke) and low - radius <= x <= high + radius:
                    return True
            else:
                coordinate = (first[0] + second[0]) / 2.0
                low, high = sorted((first[1], second[1]))
                if abs(coordinate - x) <= max(2.0, 2.0 * stroke) and low - radius <= y <= high + radius:
                    return True
        return False

    for component_index, (x1, y1, x2, y2) in enumerate(component_boxes):
        # The bbox often covers only the resistor/capacitor body while the
        # rendered red terminal stem extends roughly one body length beyond
        # it.  Scale the local search from the larger body dimension rather
        # than using a fixed 40--80 px margin; this remains small on icons and
        # sufficient on high-resolution symbols.
        body_span = max(1.0, max(x2 - x1, y2 - y1))
        reach = max(10, min(96, int(round(1.25 * body_span))))
        lateral = max(1, min(4, int(math.ceil(1.5 * stroke))))
        run_required = max(3, min(8, int(math.ceil(2.0 * stroke + 1.0))))
        sides = (
            ("left", x1, int(math.floor(y1 + lateral)), int(math.ceil(y2 - lateral))),
            ("right", x2, int(math.floor(y1 + lateral)), int(math.ceil(y2 - lateral))),
            ("top", y1, int(math.floor(x1 + lateral)), int(math.ceil(x2 - lateral))),
            ("bottom", y2, int(math.floor(x1 + lateral)), int(math.ceil(x2 - lateral))),
        )
        for side, boundary, axis_start, axis_end in sides:
            dx, dy = directions[side]
            # Keep the outward green contact together with its scan axis;
            # using the last loop coordinate here would associate a terminal
            # with an unrelated row/column in the same component ROI.
            axis_hits: list[tuple[float, int, int]] = []
            for axis_int in range(max(0, axis_start), min((height if side in {"left", "right"} else width) - 1, axis_end) + 1):
                red_near = False
                green_steps: list[int] = []
                skeleton_step = False
                for step in range(-max(2, int(math.ceil(1.5 * stroke))), reach + 1):
                    base_x = boundary + dx * step if side in {"left", "right"} else axis_int
                    base_y = axis_int if side in {"left", "right"} else boundary + dy * step
                    for offset in range(-lateral, lateral + 1):
                        xx = int(round(base_x - dy * offset))
                        yy = int(round(base_y + dx * offset))
                        if not (0 <= xx < width and 0 <= yy < height):
                            continue
                        if step <= max(1, int(math.ceil(1.5 * stroke))) and red[yy, xx]:
                            red_near = True
                        if step >= 0 and green[yy, xx]:
                            green_steps.append(step)
                            if skeleton_mask[yy, xx]:
                                skeleton_step = True
                if not red_near or not green_steps:
                    continue
                first_green = min(green_steps)
                run = 0
                gap = 0
                for step in range(first_green, min(reach, first_green + max(8, run_required + 3)) + 1):
                    base_x = boundary + dx * step if side in {"left", "right"} else axis_int
                    base_y = axis_int if side in {"left", "right"} else boundary + dy * step
                    hit = False
                    for offset in range(-lateral, lateral + 1):
                        xx = int(round(base_x - dy * offset))
                        yy = int(round(base_y + dx * offset))
                        if 0 <= xx < width and 0 <= yy < height and green[yy, xx]:
                            hit = True
                            break
                    if hit:
                        run += 1
                        gap = 0
                    else:
                        gap += 1
                        if gap > 1:
                            break
                # A terminal can turn immediately after leaving the symbol:
                # e.g. a red vertical component lead meets a horizontal green
                # wire at its first outward pixel.  In that case a strict
                # normal-direction run would reject a real terminal.  Permit
                # a local turn only when the green cross-section has enough
                # pixels and the turn is supported by the skeleton/Hough
                # evidence; isolated green decoration still fails this gate.
                turn_supported = (
                    run < run_required
                    and len(green_steps) >= max(2, int(math.ceil(stroke)))
                    and skeleton_step
                )
                if run < run_required and not turn_supported:
                    continue
                outward_x = int(round(boundary + dx * (first_green + run // 2))) if side in {"left", "right"} else int(axis_int)
                outward_y = int(axis_int) if side in {"left", "right"} else int(round(boundary + dy * (first_green + run // 2)))
                if not skeleton_step and not global_support(outward_x, outward_y, side, max(2, int(math.ceil(1.5 * stroke)))):
                    continue
                axis_hits.append((float(axis_int), outward_x, outward_y))
            if not axis_hits:
                continue
            # Collapse the several raster rows/columns of one terminal while
            # retaining separate terminals at a measured pitch.
            axis_hits.sort(key=lambda item: item[0])
            groups: list[list[tuple[float, int, int]]] = []
            for value in axis_hits:
                if not groups or value[0] - groups[-1][-1][0] > max(2.0, 2.0 * stroke):
                    groups.append([value])
                else:
                    groups[-1].append(value)
            for group in groups:
                axis = float(np.median([item[0] for item in group]))
                if any(abs(axis - old) <= max(2.0, 2.0 * stroke) for old in existing_axis.get((component_index, side), [])):
                    continue
                point = (
                    (float(boundary), axis)
                    if side in {"left", "right"}
                    else (axis, float(boundary))
                )
                skeleton_point = (
                    (
                        float(np.median([item[1] for item in group])),
                        float(np.median([item[2] for item in group])),
                    )
                )
                seeds.append(
                    {
                        "point_top_left": _as_point(point),
                        "component_boundary_reference_top_left": _as_point(point),
                        "skeleton_point_top_left": _as_point(skeleton_point),
                        "component_index": component_index,
                        "side": side,
                        "boundary_distance_px": 0.0,
                        "source": "component_boundary_red_green_transition",
                        "terminal_seed_evidence": "red_near_boundary_green_outward_run",
                        "red_green_local_evidence": True,
                    }
                )
    return seeds


def _pin_anchor_nodes(
    skeleton: np.ndarray,
    pin_anchors_top_left: Iterable[dict[str, Any]] | None,
    component_boxes: list[tuple[float, float, float, float]],
    *,
    source_mask: np.ndarray | None = None,
    red_mask: np.ndarray | None = None,
    stroke_width: float = 1.0,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Bind pin points only when a local outward path reaches them.

    A pin JSON point is an observed geometric anchor, not permission to draw
    a wire.  The old nearest-pixel rule was too brittle: a 3--6 px terminal
    gap, antialiasing, or a one-pixel diagonal could make a real wire look
    disconnected.  This implementation searches a scale-derived outward
    corridor and accepts a skeleton point only when it has forward path
    support.  Red-only/disconnected pins remain unbound.
    """
    output: dict[tuple[int, int], dict[str, Any]] = {}
    height, width = skeleton.shape[:2]
    source = source_mask if source_mask is not None else skeleton
    if source.shape != skeleton.shape:
        source = skeleton
    # Pin coordinates can fall one rendered stroke inside/outside the green
    # centerline because the upstream pin tip is measured on the symbol
    # boundary while the raster wire is measured on its anti-aliased center.
    # Allow only a short, stroke-derived backtrack; this is not a search for a
    # nearby parallel wire.
    anchor_backtrack = max(1.5, min(4.0, 1.5 * max(1.0, float(stroke_width))))

    def direction_for(raw: dict[str, Any], px: float, py: float, box: tuple[float, float, float, float] | None) -> tuple[float, float, str]:
        side = str(raw.get("side") or "").lower()
        directions = {
            "left": (-1.0, 0.0),
            "right": (1.0, 0.0),
            "top": (0.0, -1.0),
            "bottom": (0.0, 1.0),
        }
        if side in directions:
            dx, dy = directions[side]
            return dx, dy, side
        if box is not None:
            x1, y1, x2, y2 = box
            distances = {
                "left": abs(px - x1),
                "right": abs(px - x2),
                "top": abs(py - y1),
                "bottom": abs(py - y2),
            }
            side = min(distances, key=distances.get)
            dx, dy = directions[side]
            return dx, dy, side
        return 0.0, 0.0, "unknown"

    def has_forward_support(node: tuple[int, int], direction: tuple[float, float], length: float) -> bool:
        """Require connected pixels to continue away from the component."""
        dx, dy = direction
        queue = [node]
        visited = {node}
        best_projection = 0.0
        while queue and len(visited) <= 512:
            current = queue.pop(0)
            offset_x = current[0] - node[0]
            offset_y = current[1] - node[1]
            projection = offset_x * dx + offset_y * dy
            best_projection = max(best_projection, projection)
            if best_projection >= max(3.0, min(10.0, 1.5 * length)):
                return True
            for neighbor in _neighbors4(skeleton, current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        # A terminal may end at another component/pin shortly after the
        # anchor.  A non-trivial connected path is still valid evidence.
        return len(visited) >= max(4, int(round(2.0 * stroke_width)))

    def red_transition_reaches_green(
        point: tuple[float, float],
        node: tuple[int, int],
        direction: tuple[float, float],
        reach: float,
        lateral_limit: float,
    ) -> bool:
        """Check a local red-terminal -> green-wire transition.

        Red pixels are never added to the global skeleton.  They can only
        justify accepting an already observed green skeleton node when a
        short red path starts at the pin and reaches that node.  This covers
        rendered red terminal stubs without turning component outlines into
        wire edges.
        """
        if red_mask is None or not cv2.countNonZero(red_mask):
            return False
        px, py = point
        dx, dy = direction
        height_, width_ = skeleton.shape[:2]
        allowed: set[tuple[int, int]] = set()
        x_start = max(0, int(math.floor(px - lateral_limit - 2)))
        x_end = min(width_ - 1, int(math.ceil(px + dx * reach + lateral_limit + 2)))
        y_start = max(0, int(math.floor(py - lateral_limit - 2)))
        y_end = min(height_ - 1, int(math.ceil(py + dy * reach + lateral_limit + 2)))
        # Use a direction-aware bounding region for all four sides.
        x_start = max(0, int(math.floor(min(px, px + dx * reach) - lateral_limit - 2)))
        x_end = min(width_ - 1, int(math.ceil(max(px, px + dx * reach) + lateral_limit + 2)))
        y_start = max(0, int(math.floor(min(py, py + dy * reach) - lateral_limit - 2)))
        y_end = min(height_ - 1, int(math.ceil(max(py, py + dy * reach) + lateral_limit + 2)))
        for yy in range(y_start, y_end + 1):
            for xx in range(x_start, x_end + 1):
                ox, oy = float(xx) - px, float(yy) - py
                forward = ox * dx + oy * dy
                lateral = abs(ox * dy - oy * dx)
                if -anchor_backtrack <= forward <= reach and lateral <= lateral_limit + 1.0:
                    allowed.add((xx, yy))
        if not allowed:
            return False
        target_near = {
            (xx, yy)
            for yy in range(max(0, node[1] - 1), min(height_, node[1] + 2))
            for xx in range(max(0, node[0] - 1), min(width_, node[0] + 2))
            if skeleton[yy, xx]
        }
        starts = [point_ for point_ in allowed if red_mask[point_[1], point_[0]]]
        if not starts:
            return False
        queue = list(starts)
        visited = set(starts)
        while queue and len(visited) <= 256:
            current = queue.pop(0)
            if any(
                math.hypot(current[0] - target[0], current[1] - target[1]) <= 2.0
                for target in target_near
            ):
                return True
            cx, cy = current
            for nx in range(cx - 1, cx + 2):
                for ny in range(cy - 1, cy + 2):
                    if (nx, ny) not in allowed or (nx, ny) in visited:
                        continue
                    if not red_mask[ny, nx]:
                        continue
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    for raw in pin_anchors_top_left or []:
        if not isinstance(raw, dict):
            continue
        point = raw.get("point_top_left")
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (TypeError, ValueError):
            continue
        index = raw.get("component_index")
        try:
            box = component_boxes[int(index)] if index is not None else None
        except (TypeError, ValueError, IndexError):
            box = None
        dx, dy, side = direction_for(raw, px, py, box)
        if side == "unknown":
            continue
        scale = min(box[2] - box[0], box[3] - box[1]) if box else 32.0
        # The corridor is adaptive, bounded, and measured in image pixels.
        # It is deliberately wider than the immediate attachment radius so a
        # short pin stub can be recovered without accepting arbitrary nearby
        # wires.
        reach = max(8.0, min(64.0, max(0.18 * scale, 4.0 * stroke_width)))
        # Prefer the observed same-side pin pitch to the raw bbox span when
        # limiting the local corridor.  A bbox-only lane is too wide for
        # dense ICs and can jump to an adjacent pin; the pitch is used only
        # as an evidence bound and never invents a terminal.
        same_side_axes: list[float] = []
        for other in pin_anchors_top_left or []:
            if not isinstance(other, dict):
                continue
            if (
                other.get("component_index") != index
                or str(other.get("side") or "").lower() != side
            ):
                continue
            other_point = other.get("point_top_left")
            if not isinstance(other_point, dict):
                continue
            try:
                same_side_axes.append(
                    float(other_point["y"] if side in {"left", "right"} else other_point["x"])
                )
            except (KeyError, TypeError, ValueError):
                continue
        local_pitch = None
        if same_side_axes:
            current_axis = py if side in {"left", "right"} else px
            gaps = [
                abs(current_axis - value)
                for value in same_side_axes
                if abs(current_axis - value) > max(1.5, 2.0 * max(1.0, float(stroke_width)))
            ]
            if gaps:
                local_pitch = float(np.median(gaps))
        pitch_limit = 0.45 * local_pitch if local_pitch is not None else 0.06 * max(1.0, scale)
        lateral_limit = max(2.0, min(8.0, pitch_limit))
        red_lateral_limit = max(lateral_limit, min(8.0, 3.0 * stroke_width))
        attach_limit = max(4.0, min(16.0, max(0.25 * scale, 4.0 * stroke_width)))
        search_lateral = max(lateral_limit, red_lateral_limit)
        corridor_x = (px + min(0.0, dx * reach), px + max(0.0, dx * reach))
        corridor_y = (py + min(0.0, dy * reach), py + max(0.0, dy * reach))
        x0 = max(0, int(math.floor(corridor_x[0] - search_lateral - 1)))
        x1 = min(width - 1, int(math.ceil(corridor_x[1] + search_lateral + 1)))
        y0 = max(0, int(math.floor(corridor_y[0] - search_lateral - 1)))
        y1 = min(height - 1, int(math.ceil(corridor_y[1] + search_lateral + 1)))
        candidates: list[tuple[float, tuple[int, int], float, float]] = []
        used_red_transition = False
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if not skeleton[y, x]:
                    continue
                offset_x, offset_y = float(x) - px, float(y) - py
                forward = offset_x * dx + offset_y * dy
                lateral = abs(offset_x * dy - offset_y * dx)
                distance = math.hypot(offset_x, offset_y)
                if forward < -anchor_backtrack or forward > reach or lateral > lateral_limit or distance > attach_limit:
                    continue
                if not has_forward_support((x, y), (dx, dy), reach):
                    continue
                # Favor short attachment distance, then exact outward
                # alignment.  No case-specific coordinate is used.
                score = distance + 0.75 * lateral + 0.15 * max(0.0, forward)
                candidates.append((score, (x, y), distance, forward))
        # For black images only, recover a one-pixel antialiased gap from the
        # unfiltered achromatic source.  The recovered point still has to be
        # close to the skeleton and pass the same outward-path test.
        if not candidates and source is not skeleton:
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if not source[y, x]:
                        continue
                    offset_x, offset_y = float(x) - px, float(y) - py
                    forward = offset_x * dx + offset_y * dy
                    lateral = abs(offset_x * dy - offset_y * dx)
                    if forward < -anchor_backtrack or forward > reach or lateral > lateral_limit:
                        continue
                    nearby = [
                        (sx, sy)
                        for sy in range(max(0, y - 2), min(height, y + 3))
                        for sx in range(max(0, x - 2), min(width, x + 3))
                        if skeleton[sy, sx]
                    ]
                    if not nearby:
                        continue
                    node = min(nearby, key=lambda item: math.hypot(item[0] - x, item[1] - y))
                    if not has_forward_support(node, (dx, dy), reach):
                        continue
                    distance = math.hypot(float(node[0]) - px, float(node[1]) - py)
                    if distance <= attach_limit + 2.0:
                        score = distance + 0.75 * lateral + 0.15 * max(0.0, forward)
                        candidates.append((score, node, distance, forward))
        # A green wire can be separated from the pin by a short red-rendered
        # terminal.  Search green skeleton nodes farther along the same
        # corridor, but accept them only when a red path reaches them locally.
        if not candidates and red_mask is not None:
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if not skeleton[y, x]:
                        continue
                    offset_x, offset_y = float(x) - px, float(y) - py
                    forward = offset_x * dx + offset_y * dy
                    lateral = abs(offset_x * dy - offset_y * dx)
                    distance = math.hypot(offset_x, offset_y)
                    if forward < -anchor_backtrack or forward > reach or lateral > red_lateral_limit:
                        continue
                    if not has_forward_support((x, y), (dx, dy), reach):
                        continue
                    if not red_transition_reaches_green(
                        (px, py), (x, y), (dx, dy), reach, red_lateral_limit
                    ):
                        continue
                    score = distance + 0.75 * lateral + 0.15 * max(0.0, forward)
                    candidates.append((score, (x, y), distance, forward))
                    used_red_transition = True
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        _score, node, distance, _forward = candidates[0]
        if node in output:
            # Keep the closest anchor when two pin records quantise to one
            # raster node; do not create duplicate wire endpoints.
            old = output[node]
            old_distance = math.hypot(
                float(old["point_top_left"]["x"]) - node[0],
                float(old["point_top_left"]["y"]) - node[1],
            )
            if candidates[0][0] >= old_distance:
                continue
        anchor = dict(raw)
        anchor["point_top_left"] = {"x": round(px, 2), "y": round(py, 2)}
        anchor["side"] = side
        anchor["component_scale"] = round(float(scale), 3)
        anchor["skeleton_distance_px"] = round(distance, 3)
        anchor["pin_path_evidence"] = True
        if used_red_transition:
            anchor["terminal_transition"] = "red_to_green_local"
        output[node] = anchor
    return output


def _recover_pin_anchor_paths(
    skeleton: np.ndarray,
    pin_anchors_top_left: Iterable[dict[str, Any]] | None,
    existing: dict[tuple[int, int], dict[str, Any]],
    component_boxes: list[tuple[float, float, float, float]],
    source_mask: np.ndarray,
    stroke_width: float,
) -> tuple[np.ndarray, dict[tuple[int, int], dict[str, Any]], int]:
    """Recover a pin-to-skeleton gap only from local source-mask evidence.

    This is intentionally narrower than global wire completion: it starts at
    a pin anchor, follows only the outward corridor, and must reach the
    pre-existing skeleton.  No path is created for an isolated pin or from a
    library/order prior.
    """
    work = np.where(skeleton > 0, 255, 0).astype(np.uint8)
    original = work.copy()
    source = np.asarray(source_mask > 0, dtype=np.uint8)
    height, width = work.shape[:2]
    recovered = 0
    anchor_backtrack = max(1.5, min(4.0, 1.5 * max(1.0, float(stroke_width))))

    directions = {
        "left": (-1, 0),
        "right": (1, 0),
        "top": (0, -1),
        "bottom": (0, 1),
    }

    for raw in pin_anchors_top_left or []:
        if not isinstance(raw, dict):
            continue
        point = raw.get("point_top_left")
        if not isinstance(point, dict):
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(
            math.hypot(px - float(item.get("point_top_left", {}).get("x", px)),
                       py - float(item.get("point_top_left", {}).get("y", py)))
            <= max(1.0, 2.0 * stroke_width)
            for item in existing.values()
        ):
            continue
        index = raw.get("component_index")
        try:
            box = component_boxes[int(index)] if index is not None else None
        except (TypeError, ValueError, IndexError):
            box = None
        side = str(raw.get("side") or "").lower()
        if side not in directions and box is not None:
            x1, y1, x2, y2 = box
            side = min(
                {
                    "left": abs(px - x1),
                    "right": abs(px - x2),
                    "top": abs(py - y1),
                    "bottom": abs(py - y2),
                },
                key={
                    "left": abs(px - x1),
                    "right": abs(px - x2),
                    "top": abs(py - y1),
                    "bottom": abs(py - y2),
                }.get,
            )
        if side not in directions:
            continue
        dx, dy = directions[side]
        scale = min(box[2] - box[0], box[3] - box[1]) if box else 32.0
        reach = max(4.0 * stroke_width, min(64.0, 0.22 * max(1.0, scale)))
        lateral = max(1.5 * stroke_width, min(8.0, 0.08 * max(1.0, scale)))
        x0 = max(0, int(math.floor(min(px, px + dx * reach) - lateral - 1)))
        x1 = min(width - 1, int(math.ceil(max(px, px + dx * reach) + lateral + 1)))
        y0 = max(0, int(math.floor(min(py, py + dy * reach) - lateral - 1)))
        y1 = min(height - 1, int(math.ceil(max(py, py + dy * reach) + lateral + 1)))

        def allowed(x: int, y: int) -> bool:
            ox, oy = float(x) - px, float(y) - py
            forward = ox * dx + oy * dy
            side_distance = abs(ox * dy - oy * dx)
            return (
                -anchor_backtrack <= forward <= reach
                and side_distance <= lateral
                and bool(source[y, x])
            )

        starts = [
            (x, y)
            for y in range(y0, y1 + 1)
            for x in range(x0, x1 + 1)
            if allowed(x, y) and math.hypot(x - px, y - py) <= max(2.0, 2.5 * stroke_width)
        ]
        if not starts:
            continue
        queue = list(starts)
        parent: dict[tuple[int, int], tuple[int, int] | None] = {item: None for item in starts}
        target: tuple[int, int] | None = None
        while queue and len(parent) <= 512:
            current = queue.pop(0)
            if original[current[1], current[0]] and math.hypot(current[0] - px, current[1] - py) > max(2.0, 2.5 * stroke_width):
                target = current
                break
            cx, cy = current
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if not (x0 <= nx <= x1 and y0 <= ny <= y1) or (nx, ny) in parent:
                    continue
                if not allowed(nx, ny):
                    continue
                parent[(nx, ny)] = current
                queue.append((nx, ny))
        if target is None:
            continue
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = target
        while current is not None:
            path.append(current)
            current = parent.get(current)
        if len(path) < 2:
            continue
        for x, y in path:
            work[y, x] = 255
        anchor = dict(raw)
        anchor["pin_path_evidence"] = True
        anchor["terminal_transition"] = "local_source_to_skeleton"
        anchor["skeleton_distance_px"] = round(math.hypot(target[0] - px, target[1] - py), 3)
        existing[target] = anchor
        recovered += 1
    return work, existing, recovered


def _junction_records(
    skeleton: np.ndarray,
    terminal_nodes: set[tuple[int, int]],
    *,
    stroke_width: float = 1.0,
    wire_mask: np.ndarray | None = None,
    red_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Cluster branch pixels into junction candidates.

    A crossing is only a geometric candidate here.  Electrical connectivity
    is deliberately deferred until net construction, where a junction-dot
    or branch-continuity rule can decide whether paths are actually joined.
    """
    binary = (skeleton > 0).astype(np.uint8)
    degrees = _degree4_map(skeleton)
    branch = ((binary > 0) & (degrees >= 3)).astype(np.uint8) * 255
    if not cv2.countNonZero(branch):
        return []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(branch, 8)
    records: list[dict[str, Any]] = []
    for label in range(1, count):
        x, y = map(float, centroids[label])
        # A junction a few pixels beyond a pin terminal is still a junction;
        # the previous 5 px exclusion erased legitimate nearby branches.
        terminal_exclusion = max(2.0, min(3.0, 1.5 * stroke_width))
        if any(math.hypot(x - node[0], y - node[1]) <= terminal_exclusion for node in terminal_nodes):
            continue
        cx, cy = int(round(x)), int(round(y))
        patch_radius = max(2, int(round(1.5 * stroke_width)))
        patch = binary[
            max(0, cy - patch_radius):cy + patch_radius + 1,
            max(0, cx - patch_radius):cx + patch_radius + 1,
        ]
        local_density = float(patch.mean()) if patch.size else 0.0
        raw_source = binary if wire_mask is None else (wire_mask > 0).astype(np.uint8)
        raw_patch_radius = max(2, int(round(2.5 * stroke_width)))
        raw_patch = raw_source[
            max(0, cy - raw_patch_radius):cy + raw_patch_radius + 1,
            max(0, cx - raw_patch_radius):cx + raw_patch_radius + 1,
        ]
        raw_density = float(raw_patch.mean()) if raw_patch.size else 0.0
        center_patch = raw_source[
            max(0, cy - 1):cy + 2,
            max(0, cx - 1):cx + 2,
        ]
        center_density = float(center_patch.mean()) if center_patch.size else 0.0
        red_center_density = 0.0
        red_patch_density = 0.0
        if red_mask is not None and red_mask.shape == skeleton.shape:
            red_binary = (red_mask > 0).astype(np.uint8)
            red_patch = red_binary[
                max(0, cy - raw_patch_radius):cy + raw_patch_radius + 1,
                max(0, cx - raw_patch_radius):cx + raw_patch_radius + 1,
            ]
            red_center = red_binary[
                max(0, cy - 1):cy + 2,
                max(0, cx - 1):cx + 2,
            ]
            red_patch_density = float(red_patch.mean()) if red_patch.size else 0.0
            red_center_density = float(red_center.mean()) if red_center.size else 0.0
        branch_degree = int(degrees[min(max(cy, 0), degrees.shape[0] - 1), min(max(cx, 0), degrees.shape[1] - 1)])
        direction_support = 0
        ray_length = max(2, int(round(3.0 * stroke_width)))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            supported = False
            for step in range(1, ray_length + 1):
                rx, ry = cx + dx * step, cy + dy * step
                if 0 <= rx < binary.shape[1] and 0 <= ry < binary.shape[0] and binary[ry, rx]:
                    supported = True
                    break
            direction_support += int(supported)
        # Count opposite directional pairs from the same local ray evidence.
        def ray_has(dx: int, dy: int) -> bool:
            for step in range(1, ray_length + 1):
                rx, ry = cx + dx * step, cy + dy * step
                if not (0 <= rx < binary.shape[1] and 0 <= ry < binary.shape[0]):
                    break
                if binary[ry, rx]:
                    return True
            return False

        horizontal_pair = int(ray_has(-1, 0) and ray_has(1, 0))
        vertical_pair = int(ray_has(0, -1) and ray_has(0, 1))
        opposite_pair_count = horizontal_pair + vertical_pair
        # Dot evidence is intentionally conservative.  The skeleton alone
        # cannot distinguish a crossing from a connected four-way node; use
        # the original wire mask and the branch-cluster area as extra evidence.
        dot_evidence = bool(
            direction_support >= 4
            and int(stats[label, cv2.CC_STAT_AREA]) >= max(2, int(round(stroke_width)))
            and (center_density >= 0.55 or red_center_density >= 0.35)
            and (raw_density >= 0.30 or red_patch_density >= 0.12)
        )
        # A four-ray crossing is not automatically an electrical junction.
        # A three-ray branch remains a join candidate only when the original
        # mask supports the centre; otherwise it is left for validation.
        if direction_support >= 4:
            classification = "junction_candidate" if dot_evidence else "crossing_candidate"
            electrical_join = "join_candidate" if dot_evidence else "do_not_join_without_dot_evidence"
        elif direction_support == 3:
            classification = "junction_candidate" if max(center_density, red_center_density) >= 0.35 else "branch_candidate"
            electrical_join = "join_candidate" if classification == "junction_candidate" else "requires_topology_validation"
        else:
            classification = "branch_candidate"
            electrical_join = "requires_topology_validation"
        records.append(
            {
                "point_top_left": _as_point((x, y)),
                "pixel_count": int(stats[label, cv2.CC_STAT_AREA]),
                "branch_degree": branch_degree,
                "direction_support": direction_support,
                "opposite_pair_count": opposite_pair_count,
                "local_density": round(local_density, 4),
                "raw_local_density": round(raw_density, 4),
                "center_density": round(center_density, 4),
                "red_center_density": round(red_center_density, 4),
                "red_patch_density": round(red_patch_density, 4),
                "dot_evidence": dot_evidence,
                "classification": classification,
                "source": "skeleton_branch_candidate",
                "electrical_join": electrical_join,
            }
        )
    return records


def _snap_segment_terminals(
    segments: list[dict[str, Any]],
    terminal_records: list[dict[str, Any]],
    *,
    tolerance: float = 7.0,
) -> list[dict[str, Any]]:
    """Move only segment endpoints near a terminal onto the real boundary."""
    output: list[dict[str, Any]] = []
    terminal_points = [
        (record["point_top_left"], record)
        for record in terminal_records
        if isinstance(record.get("point_top_left"), dict)
    ]
    for segment in segments:
        item = dict(segment)
        # A component red->green transition already contains its measured
        # red outer endpoint and green contact endpoint.  Snapping it to the
        # nearest coarse bbox terminal would erase exactly the local geometry
        # this candidate was introduced to preserve.
        if item.get("local_red_green_evidence"):
            output.append(item)
            continue
        points = []
        endpoint_types: list[str] = []
        terminal_bindings: list[dict[str, Any] | None] = []
        previous_types = segment.get("endpoint_types", [])
        previous_bindings = segment.get("terminal_bindings", [])
        for point_index, raw in enumerate(segment.get("points_top_left", [])):
            point = _point(raw)
            nearest = None
            nearest_distance = float("inf")
            for terminal_point, record in terminal_points:
                candidate = _point(terminal_point)
                distance = math.hypot(point[0] - candidate[0], point[1] - candidate[1])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = record
            if nearest is not None and nearest_distance <= tolerance:
                points.append(nearest["point_top_left"])
                endpoint_types.append("component_terminal")
                terminal_bindings.append(
                    {
                        key: nearest[key]
                        for key in ("component_key", "pin_key", "component_index", "side")
                        if nearest.get(key) is not None
                    }
                )
            else:
                points.append(_as_point(point))
                previous_type = (
                    previous_types[point_index]
                    if isinstance(previous_types, list) and point_index < len(previous_types)
                    else None
                )
                previous_binding = (
                    previous_bindings[point_index]
                    if isinstance(previous_bindings, list) and point_index < len(previous_bindings)
                    else None
                )
                endpoint_types.append(str(previous_type or "wire_node"))
                terminal_bindings.append(previous_binding if isinstance(previous_binding, dict) else None)
        if len(points) == 2:
            item["points_top_left"] = points
            item["endpoint_types"] = endpoint_types
            item["terminal_bindings"] = terminal_bindings
        output.append(item)
    return output


def _snap_segment_junctions(
    segments: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Snap edge endpoints to the centre of an observed branch cluster."""
    if not segments or not junctions:
        return segments
    radius = max(2.0, min(8.0, 2.5 * max(1.0, float(stroke_width))))
    candidates = [
        record for record in junctions
        if record.get("classification") == "junction_candidate"
        and isinstance(record.get("point_top_left"), dict)
    ]
    output: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        points = []
        endpoint_types = list(item.get("endpoint_types", []))
        if len(endpoint_types) != 2:
            endpoint_types = ["wire_node", "wire_node"]
        for index, raw_point in enumerate(item.get("points_top_left", [])):
            point = _point(raw_point)
            nearest = None
            nearest_distance = float("inf")
            for record in candidates:
                candidate = _point(record["point_top_left"])
                distance = math.hypot(point[0] - candidate[0], point[1] - candidate[1])
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest = candidate
            if nearest is not None and nearest_distance <= radius:
                points.append(_as_point(nearest))
                endpoint_types[index] = "junction_node"
            else:
                points.append(_as_point(point))
        if len(points) == 2:
            item["points_top_left"] = points
            item["endpoint_types"] = endpoint_types
        output.append(item)
    return output


def _snap_bound_hough_terminals(
    segments: list[dict[str, Any]],
    terminal_records: list[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Project only bound global-Hough endpoints onto their real pin tip.

    A Hough interval is allowed to own the wire geometry, but its endpoint
    can stop a few raster pixels inside the red/green terminal transition.
    This correction is intentionally narrower than generic terminal snapping:

    * the segment must already carry the same explicit component/pin binding;
    * the orientation must agree with the terminal side; and
    * only ``wire_skeleton_validated_hough`` is changed, so local candidates
      are not duplicated or moved by a nearby unrelated pin.

    No segment is created and no unbound endpoint is guessed.
    """
    if not segments or not terminal_records:
        return segments
    lookup: dict[tuple[Any, Any], dict[str, Any]] = {}
    for record in terminal_records:
        if not record.get("pin_anchor_evidence", False):
            continue
        point = record.get("point_top_left")
        if not isinstance(point, dict):
            continue
        key = (record.get("component_key"), record.get("pin_key"))
        if key[0] is None or key[1] is None:
            continue
        lookup[key] = record
    if not lookup:
        return segments
    output: list[dict[str, Any]] = []
    stroke = max(1.0, float(stroke_width))
    for segment in segments:
        item = dict(segment)
        if item.get("source") != "wire_skeleton_validated_hough":
            output.append(item)
            continue
        points = item.get("points_top_left")
        bindings = item.get("terminal_bindings")
        if not isinstance(points, list) or len(points) != 2 or not isinstance(bindings, list):
            output.append(item)
            continue
        points = [dict(point) if isinstance(point, dict) else point for point in points]
        for endpoint_index in range(2):
            binding = bindings[endpoint_index] if endpoint_index < len(bindings) else None
            if not isinstance(binding, dict):
                continue
            record = lookup.get((binding.get("component_key"), binding.get("pin_key")))
            if record is None or not isinstance(points[endpoint_index], dict):
                continue
            side = str(record.get("side") or "").lower()
            orientation = str(item.get("orientation") or "")
            compatible = (
                orientation == "horizontal" and side in {"left", "right"}
            ) or (
                orientation == "vertical" and side in {"top", "bottom"}
            )
            if not compatible:
                continue
            target = record.get("point_top_left")
            if not isinstance(target, dict):
                continue
            try:
                px, py = float(points[endpoint_index]["x"]), float(points[endpoint_index]["y"])
                tx, ty = float(target["x"]), float(target["y"])
            except (KeyError, TypeError, ValueError):
                continue
            axis_error = abs(py - ty) if orientation == "horizontal" else abs(px - tx)
            distance = math.hypot(px - tx, py - ty)
            scale = max(1.0, float(record.get("component_scale") or 32.0))
            radius = max(3.0, min(0.35 * scale, 4.0 * stroke))
            if distance <= radius and axis_error <= max(2.0, 1.5 * stroke):
                points[endpoint_index] = _as_point((tx, ty))
                item["bound_terminal_projection"] = True
        item["points_top_left"] = points
        output.append(item)
    return output


def _refine_axis_coordinate_from_mask(
    segments: list[dict[str, Any]],
    support_mask: np.ndarray,
    *,
    stroke_width: float,
) -> tuple[list[dict[str, Any]], int]:
    """Refine the raster centre line of global axis-aligned candidates.

    HoughLinesP may return the correct wire interval one raster row/column
    away from the rendered centre, especially after anti-aliasing and
    downsampling.  That one-pixel bias is enough to fail a strict endpoint
    match even when both endpoints are otherwise correct.  This operation
    therefore changes only the perpendicular axis of an already accepted
    image-supported candidate:

    * only global Hough/skeleton observations are refined;
    * the search radius is derived from the measured stroke width;
    * the longitudinal endpoints are never extended or moved;
    * ties prefer the existing axis, so nearby parallel wires are not
      selected by a fixed image-wide coordinate.

    Red terminal pixels are deliberately not passed as ``support_mask``.
    They remain local transition evidence and cannot move a global wire.
    """
    if not segments or support_mask is None or support_mask.size == 0:
        return segments, 0
    height, width = support_mask.shape[:2]
    radius = max(1, min(6, int(math.ceil(max(1.0, float(stroke_width))))))
    output: list[dict[str, Any]] = []
    changed = 0
    for raw in segments:
        item = dict(raw)
        source = str(item.get("source") or "")
        # Local fused candidates are also eligible: their red part supplies
        # the terminal transition, while the green part supplies the centre
        # line.  A pure-red observation has no green support and is left
        # untouched by the support check below.
        if source in {"pin_local_red_terminal", "pin_local_red_terminal_path"} or item.get("orientation") not in {"horizontal", "vertical"}:
            output.append(item)
            continue
        try:
            first, second = _segment_points(item)
        except (KeyError, TypeError, ValueError):
            output.append(item)
            continue
        if item["orientation"] == "horizontal":
            start, end = sorted((int(round(first[0])), int(round(second[0]))))
            coordinate = (first[1] + second[1]) / 2.0
            start = max(0, min(width - 1, start))
            end = max(0, min(width - 1, end))
            if end <= start:
                output.append(item)
                continue
            candidates: list[tuple[int, int]] = []
            for row in range(
                max(0, int(round(coordinate)) - radius),
                min(height, int(round(coordinate)) + radius + 1),
            ):
                support = int(cv2.countNonZero(support_mask[row, start : end + 1]))
                candidates.append((support, row))
            if not candidates:
                output.append(item)
                continue
            best_support = max(item[0] for item in candidates)
            best_rows = [row for support, row in candidates if support == best_support]
            best_row = float(np.median(np.asarray(best_rows, dtype=np.float32)))
            # A candidate with no longitudinal support is not made more
            # authoritative by a nearby row; retain its original geometry.
            if best_support <= 0:
                output.append(item)
                continue
            if abs(float(best_row) - coordinate) > 0.25:
                item["points_top_left"] = [
                    _as_point((first[0], float(best_row))),
                    _as_point((second[0], float(best_row))),
                ]
                changed += 1
        else:
            start, end = sorted((int(round(first[1])), int(round(second[1]))))
            coordinate = (first[0] + second[0]) / 2.0
            start = max(0, min(height - 1, start))
            end = max(0, min(height - 1, end))
            if end <= start:
                output.append(item)
                continue
            candidates: list[tuple[int, int]] = []
            for column in range(
                max(0, int(round(coordinate)) - radius),
                min(width, int(round(coordinate)) + radius + 1),
            ):
                support = int(cv2.countNonZero(support_mask[start : end + 1, column]))
                candidates.append((support, column))
            if not candidates:
                output.append(item)
                continue
            best_support = max(item[0] for item in candidates)
            best_columns = [column for support, column in candidates if support == best_support]
            best_column = float(np.median(np.asarray(best_columns, dtype=np.float32)))
            if best_support <= 0:
                output.append(item)
                continue
            if abs(float(best_column) - coordinate) > 0.25:
                item["points_top_left"] = [
                    _as_point((float(best_column), first[1])),
                    _as_point((float(best_column), second[1])),
                ]
                changed += 1
        output.append(item)
    return output, changed


def _snap_segment_skeleton_nodes(
    segments: list[dict[str, Any]],
    skeleton: np.ndarray,
    *,
    stroke_width: float,
    evidence_points: Iterable[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Snap endpoints to nearby measured skeleton endpoints/branch pixels.

    Hough endpoints can stop inside a continuous raster run.  Only degree-1
    and degree>=3 pixels are eligible here; degree-2 pixels are deliberately
    excluded so this cannot extend a wire to an arbitrary interior pixel.
    The search radius and axis tolerance follow the measured stroke width.
    """
    if not segments or skeleton.size == 0:
        return segments
    degree = _degree4_map(skeleton)
    node_mask = np.where((skeleton > 0) & (degree == 1), 255, 0).astype(np.uint8)
    if not cv2.countNonZero(node_mask):
        return segments
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(node_mask, connectivity=8)
    component_count, component_labels, _component_stats, _component_centroids = cv2.connectedComponentsWithStats(
        (skeleton > 0).astype(np.uint8), connectivity=4
    )
    evidence_labels: set[int] = set()
    evidence_radius = max(2, int(round(2.5 * max(1.0, float(stroke_width)))))
    for raw_x, raw_y in evidence_points or []:
        cx, cy = int(round(raw_x)), int(round(raw_y))
        y0, y1 = max(0, cy - evidence_radius), min(skeleton.shape[0], cy + evidence_radius + 1)
        x0, x1 = max(0, cx - evidence_radius), min(skeleton.shape[1], cx + evidence_radius + 1)
        values = np.unique(component_labels[y0:y1, x0:x1])
        evidence_labels.update(int(value) for value in values if int(value) > 0)
    nodes: list[tuple[float, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        cx, cy = centroids[label]
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        if evidence_labels:
            component_values = {
                int(component_labels[int(y), int(x)])
                for x, y in zip(xs.tolist(), ys.tolist())
                if 0 <= int(y) < component_labels.shape[0] and 0 <= int(x) < component_labels.shape[1]
            }
            if not component_values.intersection(evidence_labels):
                continue
        index = int(np.argmin((xs.astype(np.float32) - float(cx)) ** 2 + (ys.astype(np.float32) - float(cy)) ** 2))
        nodes.append((float(xs[index]), float(ys[index])))
    radius = max(2.0, min(6.0, 2.5 * max(1.0, float(stroke_width))))
    axis_tolerance = max(1.5, min(4.0, 1.25 * max(1.0, float(stroke_width))))
    output: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        if item.get("local_red_green_evidence"):
            output.append(item)
            continue
        try:
            first, second = _segment_points(item)
        except (KeyError, TypeError, ValueError):
            output.append(item)
            continue
        horizontal = abs(second[0] - first[0]) >= abs(second[1] - first[1])
        points = [first, second]
        for index, point in enumerate(points):
            best: tuple[float, tuple[float, float]] | None = None
            for node in nodes:
                if horizontal and abs(node[1] - point[1]) > axis_tolerance:
                    continue
                if not horizontal and abs(node[0] - point[0]) > axis_tolerance:
                    continue
                distance = math.hypot(point[0] - node[0], point[1] - node[1])
                if distance <= radius and (best is None or distance < best[0]):
                    best = (distance, node)
            if best is not None:
                points[index] = best[1]
        item["points_top_left"] = [_as_point(points[0]), _as_point(points[1])]
        output.append(item)
    return output


def _has_skeleton_support(
    segment: dict[str, Any],
    skeleton: np.ndarray,
    *,
    stroke_width: float,
) -> bool:
    """Require a line candidate to follow the observed 4-neighbour skeleton."""
    try:
        p1, p2 = _segment_points(segment)
    except (KeyError, TypeError, ValueError):
        return False
    length = max(1.0, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    samples = max(3, int(round(2.0 * length)))
    radius = max(1, min(4, int(round(1.25 * max(1.0, float(stroke_width))))))
    supported = 0
    height, width = skeleton.shape[:2]
    for index in range(samples + 1):
        ratio = index / float(samples)
        x = int(round(p1[0] + ratio * (p2[0] - p1[0])))
        y = int(round(p1[1] + ratio * (p2[1] - p1[1])))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        supported += int(cv2.countNonZero(skeleton[y0:y1, x0:x1]) > 0)
    # A short antialiased gap is allowed, but a candidate with no continuous
    # skeleton evidence is rejected rather than emitted as a Hough-only wire.
    # One-pixel antialias gaps are common after 4-neighbour thinning.  The
    # candidate is still generated from the green wire mask and remains
    # bounded by the measured stroke radius; a slightly lower support ratio
    # recovers valid long runs without admitting an unsupported Hough line.
    return supported / float(samples + 1) >= 0.45


def _short_skeleton_wire_candidates(
    skeleton_segments: list[dict[str, Any]],
    skeleton: np.ndarray,
    *,
    stroke_width: float,
    evidence_records: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return conservative short-path supplements for the colour backend.

    The morphology/Hough branch intentionally favours precision and therefore
    misses very short true edges.  A raw skeleton contains those edges, but it
    also contains many small branch/crossing fragments.  Keep only paths whose
    length is comparable to the measured raster stroke and whose two observed
    endpoints are not branch pixels.  This is an image-derived acceptance rule
    and does not assume a case-specific edge count or target geometry.
    """
    if not skeleton_segments:
        return []
    degree = _degree4_map(skeleton)
    maximum_length = max(3.0, min(8.0, 3.0 * max(1.0, float(stroke_width))))
    output: list[dict[str, Any]] = []
    height, width = degree.shape[:2]
    for segment in skeleton_segments:
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        if not (0.5 < length < maximum_length):
            continue
        endpoint_degrees: list[int] = []
        for point in (first, second):
            x = min(max(int(round(point[0])), 0), width - 1)
            y = min(max(int(round(point[1])), 0), height - 1)
            endpoint_degrees.append(int(degree[y, x]))
        if max(endpoint_degrees) > 2:
            continue
        item = dict(segment)
        item["source"] = "wire_skeleton_short_evidence"
        item["short_path_length_px"] = round(float(length), 3)
        item["endpoint_degrees"] = endpoint_degrees
        output.append(item)
    return output


def _pin_local_skeleton_segments(
    skeleton: np.ndarray,
    pin_anchor_by_node: dict[tuple[int, int], dict[str, Any]],
    *,
    stroke_width: float,
    component_boxes: list[tuple[float, float, float, float]] | None = None,
    max_length: float | None = None,
) -> list[dict[str, Any]]:
    """Trace only the skeleton path leaving a validated pin terminal.

    The global colour backend deliberately uses Hough as its main segment
    generator.  That is good for long runs, but it loses the exact split at a
    pin, a bend, or the first branch node.  This helper recovers that missing
    local part without skeletonizing the red component layer globally:

    * the start is an already validated ``pin_anchor_by_node`` node;
    * the first step must point away from the component side;
    * traversal is 4-neighbour and stops at the first branch/end node;
    * no path is emitted when the traversal cannot reach a real skeleton
      node, so an isolated pin cannot invent a wire.

    The path is therefore a local geometric supplement, not a library/order
    prior and not a case-specific edge-count rule.
    """
    if not pin_anchor_by_node or not cv2.countNonZero(skeleton):
        return []
    height, width = skeleton.shape[:2]
    degree = _degree4_map(skeleton)
    directions = {
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "top": (0.0, -1.0),
        "bottom": (0.0, 1.0),
    }
    limit_default = max(16.0, min(256.0, 12.0 * max(1.0, float(stroke_width))))
    if max_length is not None:
        limit_default = max(8.0, float(max_length))
    output: list[dict[str, Any]] = []

    for node, anchor in pin_anchor_by_node.items():
        side = str(anchor.get("side") or "").lower()
        direction = directions.get(side)
        if direction is None:
            continue
        dx, dy = direction
        nx, ny = node
        if not (0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]):
            continue
        scale = 32.0
        index = anchor.get("component_index")
        try:
            if component_boxes is not None and index is not None:
                box = component_boxes[int(index)]
                scale = min(box[2] - box[0], box[3] - box[1])
        except (TypeError, ValueError, IndexError):
            pass
        # The path-search radius grows with the local component scale, while
        # remaining bounded.  This allows a pin-to-bend path to reach its
        # first real graph endpoint without turning the whole image into a
        # skeleton search.
        local_limit = max(
            limit_default,
            min(64.0, 0.45 * max(1.0, float(scale))),
        )
        neighbours = _neighbors4(skeleton, node)
        if not neighbours:
            continue
        # A terminal may be one raster pixel inside the formal point.  Select
        # only neighbours with outward projection; if antialiasing makes the
        # projection tie, the closest-to-outward neighbour wins.
        ranked: list[tuple[float, float, tuple[int, int]]] = []
        for candidate in neighbours:
            vx, vy = candidate[0] - nx, candidate[1] - ny
            projection = vx * dx + vy * dy
            lateral = abs(vx * dy - vy * dx)
            ranked.append((projection, -lateral, candidate))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if ranked[0][0] <= 0.0:
            continue
        starts = [item[2] for item in ranked if item[0] > 0.0 and item[0] >= ranked[0][0] - 0.5]
        for first in starts[:1]:
            path: list[tuple[int, int]] = [node, first]
            previous, current = node, first
            visited_edges = {tuple(sorted((node, first)))}
            travelled = 1.0
            while travelled <= local_limit:
                cx, cy = current
                current_degree = int(degree[cy, cx])
                if current != first and current_degree != 2:
                    break
                next_points = [item for item in _neighbors4(skeleton, current) if item != previous]
                next_points = [
                    item for item in next_points
                    if tuple(sorted((current, item))) not in visited_edges
                ]
                if not next_points:
                    break
                # At a branch, stop at the branch node.  We do not choose one
                # branch using an arbitrary angle; the junction is the valid
                # local endpoint and the net stage handles its topology.
                if len(next_points) > 1:
                    break
                following = next_points[0]
                visited_edges.add(tuple(sorted((current, following))))
                path.append(following)
                previous, current = current, following
                travelled += math.hypot(current[0] - previous[0], current[1] - previous[1])
            if len(path) < 2:
                continue
            # A one-pixel stub that ends immediately at a pin/body boundary is
            # not enough independent evidence.  It is retained only when the
            # local path has a non-trivial raster run.
            if travelled < max(1.5, 1.25 * max(1.0, float(stroke_width))):
                continue
            # A branch is a valid graph split.  A degree-one endpoint is also
            # eligible, but only for a short path below the global Hough
            # scale; this recovers genuine pin stubs that Hough cannot see.
            end_degree = int(degree[current[1], current[0]])
            if end_degree != 1 and end_degree < 3:
                continue
            path_segments = _path_to_segments(path)
            if not path_segments:
                continue
            first_length = math.hypot(
                path_segments[0][1][0] - path_segments[0][0][0],
                path_segments[0][1][1] - path_segments[0][0][1],
            )
            endpoint_limit = max(8.0, min(local_limit, 0.25 * max(1.0, float(scale))))
            if first_length > endpoint_limit:
                continue
            # Export every observed interval up to the first real bend,
            # endpoint or branch.  A pin-to-bend-to-junction path therefore
            # contributes all of its geometric edges; dropping everything
            # after the first interval was a systematic recall loss.  The
            # traversal is already bounded by ``local_limit`` and stops at a
            # skeleton graph node, so this does not create a free extension.
            selected_segments = path_segments
            for left, right in selected_segments:
                if math.hypot(right[0] - left[0], right[1] - left[1]) < 0.75:
                    continue
                # The formal pin point is authoritative for the first path
                # endpoint; the rest remains the observed skeleton geometry.
                if math.hypot(left[0] - node[0], left[1] - node[1]) <= 1.5:
                    point = anchor.get("point_top_left")
                    if isinstance(point, dict) and "x" in point and "y" in point:
                        left = (float(point["x"]), float(point["y"]))
                orientation = (
                    "horizontal"
                    if abs(right[1] - left[1]) <= 3.0
                    else "vertical"
                    if abs(right[0] - left[0]) <= 3.0
                    else "diagonal"
                )
                output.append(
                    {
                        "points_top_left": [_as_point(left), _as_point(right)],
                        "orientation": orientation,
                        "source": "pin_local_skeleton_path",
                        "confidence": 1.0,
                        "pin_local_path_evidence": True,
                        "pin_terminal_source": "validated_pin_anchor",
                        "endpoint_types": ["component_terminal", "wire_skeleton_node"],
                        "terminal_bindings": [
                            {
                                key: anchor[key]
                                for key in ("component_key", "pin_key", "component_index", "side")
                                if anchor.get(key) is not None
                            },
                            None,
                        ],
                    }
                )
            break
    return output


def _merge_pin_local_paths(
    segments: list[dict[str, Any]],
    local_candidates: list[dict[str, Any]],
    *,
    stroke_width: float,
    append_unmatched: bool = True,
) -> list[dict[str, Any]]:
    """Merge validated pin-origin paths into nearby Hough runs.

    A local path often overlaps the first few pixels of a Hough run.  Treating
    that as a duplicate loses the authoritative pin endpoint; emitting both
    creates a duplicate edge.  This operation takes the union only when the
    two image-supported runs are collinear, close in axis coordinate, and
    overlap/touch within a stroke-derived gap.  It never joins diagonal or
    parallel-but-separated wires.
    """
    if not local_candidates:
        return segments
    output = [dict(segment) for segment in segments]
    coordinate_tolerance = max(1.5, min(4.0, 1.5 * max(1.0, float(stroke_width))))
    # Local terminal rasterization can leave a short red/green transition
    # gap between the global green run and the boundary point.  The allowed
    # gap scales with the measured stroke, and is bounded so unrelated
    # parallel tracks cannot be bridged.
    gap_tolerance = max(2.0, min(6.0, 3.0 * max(1.0, float(stroke_width))))
    for local in local_candidates:
        try:
            lp1, lp2 = _segment_points(local)
        except (KeyError, TypeError, ValueError):
            continue
        if local.get("orientation") not in {"horizontal", "vertical"}:
            output.append(local)
            continue
        orientation = str(local["orientation"])
        if orientation == "horizontal":
            local_coordinate = (lp1[1] + lp2[1]) / 2.0
            local_start, local_end = sorted((lp1[0], lp2[0]))
        else:
            local_coordinate = (lp1[0] + lp2[0]) / 2.0
            local_start, local_end = sorted((lp1[1], lp2[1]))
        best_index: int | None = None
        best_distance = float("inf")
        for index, existing in enumerate(output):
            if existing.get("orientation") != orientation:
                continue
            try:
                ep1, ep2 = _segment_points(existing)
            except (KeyError, TypeError, ValueError):
                continue
            if orientation == "horizontal":
                existing_coordinate = (ep1[1] + ep2[1]) / 2.0
                existing_start, existing_end = sorted((ep1[0], ep2[0]))
            else:
                existing_coordinate = (ep1[0] + ep2[0]) / 2.0
                existing_start, existing_end = sorted((ep1[1], ep2[1]))
            if abs(existing_coordinate - local_coordinate) > coordinate_tolerance:
                continue
            if local_start > existing_end + gap_tolerance or existing_start > local_end + gap_tolerance:
                continue
            distance = abs(existing_coordinate - local_coordinate) + abs(
                max(local_start, existing_start) - min(local_end, existing_end)
            )
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            if append_unmatched:
                output.append(local)
            continue
        existing = output[best_index]
        ep1, ep2 = _segment_points(existing)
        if orientation == "horizontal":
            coordinate = (local_coordinate + (ep1[1] + ep2[1]) / 2.0) / 2.0
            start = min(local_start, min(ep1[0], ep2[0]))
            end = max(local_end, max(ep1[0], ep2[0]))
            points = [_as_point((start, coordinate)), _as_point((end, coordinate))]
        else:
            coordinate = (local_coordinate + (ep1[0] + ep2[0]) / 2.0) / 2.0
            start = min(local_start, min(ep1[1], ep2[1]))
            end = max(local_end, max(ep1[1], ep2[1]))
            points = [_as_point((coordinate, start)), _as_point((coordinate, end))]
        merged = dict(existing)
        merged["points_top_left"] = points
        merged["source"] = (
            "non_ic_red_body_local_path"
            if local.get("source") == "red_body_local_terminal_path"
            else "wire_hough_with_pin_local_path"
        )
        merged["pin_local_path_evidence"] = True
        merged["endpoint_types"] = list(dict.fromkeys(
            list(existing.get("endpoint_types", []))
            + list(local.get("endpoint_types", []))
        ))
        merged["terminal_bindings"] = local.get("terminal_bindings", existing.get("terminal_bindings"))
        output[best_index] = merged
    return output


def _replace_hough_terminal_overlap(
    segments: list[dict[str, Any]],
    local_candidates: list[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Replace a Hough endpoint overlap with an observed terminal transition.

    A long Hough line often already contains the short interval from a real
    pin terminal to the first green pixel.  Appending the local transition
    would duplicate that interval; merging it would erase the terminal edge.
    This helper partitions the existing line at the measured transition end,
    keeps the far Hough portion, and keeps the local red/green portion.  It
    never extends beyond either observed candidate.
    """
    if not segments or not local_candidates:
        return segments
    endpoint_radius = max(2.0, min(8.0, 2.5 * max(1.0, float(stroke_width))))
    coordinate_tolerance = max(2.0, min(6.0, 2.5 * max(1.0, float(stroke_width))))
    output = [dict(item) for item in segments]
    for local in local_candidates:
        points = local.get("points_top_left")
        if not isinstance(points, list) or len(points) != 2:
            continue
        try:
            terminal = _point(points[0])
            transition = _point(points[1])
        except (KeyError, TypeError, ValueError):
            continue
        orientation = str(local.get("orientation") or "")
        if orientation not in {"horizontal", "vertical"}:
            continue
        transition_axis = transition[0] if orientation == "horizontal" else transition[1]
        terminal_axis = terminal[0] if orientation == "horizontal" else terminal[1]
        if abs(transition_axis - terminal_axis) < 0.75:
            continue
        best_index: int | None = None
        best_distance = float("inf")
        for index, existing in enumerate(output):
            if existing.get("orientation") != orientation:
                continue
            try:
                first, second = _segment_points(existing)
            except (KeyError, TypeError, ValueError):
                continue
            coordinate = (
                (first[1] + second[1]) / 2.0
                if orientation == "horizontal"
                else (first[0] + second[0]) / 2.0
            )
            local_coordinate = terminal[1] if orientation == "horizontal" else terminal[0]
            if abs(coordinate - local_coordinate) > coordinate_tolerance:
                continue
            low, high = sorted(
                (first[0], second[0]) if orientation == "horizontal" else (first[1], second[1])
            )
            if not (low - endpoint_radius <= transition_axis <= high + endpoint_radius):
                continue
            terminal_axis_clamped = min(high, max(low, terminal_axis))
            if min(abs(terminal_axis_clamped - low), abs(terminal_axis_clamped - high)) > endpoint_radius:
                continue
            distance = abs(coordinate - local_coordinate) + abs(terminal_axis_clamped - terminal_axis)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            continue
        existing = output[best_index]
        first, second = _segment_points(existing)
        low, high = sorted(
            (first[0], second[0]) if orientation == "horizontal" else (first[1], second[1])
        )
        transition_axis = min(high, max(low, transition_axis))
        coordinate = (
            (first[1] + second[1]) / 2.0
            if orientation == "horizontal"
            else (first[0] + second[0]) / 2.0
        )
        if orientation == "horizontal":
            far_start, far_end = low, high
            if abs(terminal_axis - low) <= abs(terminal_axis - high):
                far_start, far_end = transition_axis, high
            else:
                far_start, far_end = low, transition_axis
            if far_end - far_start >= 0.75:
                replacement = dict(existing)
                replacement["points_top_left"] = [
                    _as_point((far_start, coordinate)),
                    _as_point((far_end, coordinate)),
                ]
                replacement["terminal_transition_partition"] = True
                output[best_index] = replacement
            else:
                output.pop(best_index)
        else:
            far_start, far_end = low, high
            if abs(terminal_axis - low) <= abs(terminal_axis - high):
                far_start, far_end = transition_axis, high
            else:
                far_start, far_end = low, transition_axis
            if far_end - far_start >= 0.75:
                replacement = dict(existing)
                replacement["points_top_left"] = [
                    _as_point((coordinate, far_start)),
                    _as_point((coordinate, far_end)),
                ]
                replacement["terminal_transition_partition"] = True
                output[best_index] = replacement
            else:
                output.pop(best_index)
        # The Hough child now covers the measured global track on the far
        # side of the transition.  Keep the local observation as metadata on
        # that child rather than emitting a second overlapping protocol edge;
        # otherwise one physical terminal corridor is counted twice.  A
        # standalone local edge is still emitted later only when no global
        # track could be partitioned.
        replacement = output[best_index] if best_index is not None and best_index < len(output) else None
        if isinstance(replacement, dict):
            replacement["local_transition_evidence"] = True
            replacement["local_transition_points_top_left"] = [
                _as_point(terminal),
                _as_point(transition),
            ]
    return output


def _deduplicate_endpoint_equivalent_segments(
    segments: list[dict[str, Any]],
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Remove duplicate observations without merging adjacent physical edges.

    Local terminal recovery and the global skeleton can observe the same
    short edge.  They are duplicates only when both endpoints agree (in
    either direction); sharing one endpoint is not enough because that is a
    legitimate junction split.
    """
    output: list[dict[str, Any]] = []
    for candidate in segments:
        try:
            first, second = _segment_points(candidate)
        except (KeyError, TypeError, ValueError):
            continue
        duplicate = False
        for existing in output:
            try:
                left, right = _segment_points(existing)
            except (KeyError, TypeError, ValueError):
                continue
            direct = max(
                math.hypot(first[0] - left[0], first[1] - left[1]),
                math.hypot(second[0] - right[0], second[1] - right[1]),
            )
            reverse = max(
                math.hypot(first[0] - right[0], first[1] - right[1]),
                math.hypot(second[0] - left[0], second[1] - left[1]),
            )
            if min(direct, reverse) <= tolerance:
                duplicate = True
                break
        if not duplicate:
            output.append(candidate)
    return output


def _deduplicate_collinear_overlaps(
    segments: list[dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Remove repeated observations of one already measured wire interval.

    Hough, the skeleton graph and a local terminal traversal can all observe
    the same physical run.  Endpoint-only de-duplication is intentionally
    insufficient because raster quantisation gives these observations
    slightly different endpoints.  This helper removes only *high-overlap*
    collinear candidates after graph-node splitting:

    * orientation and measured axis must agree within the adaptive stroke
      tolerance;
    * the intervals must overlap almost completely;
    * touching/adjacent intervals are never merged; and
    * the stronger image-supported observation is retained.

    It therefore cannot bridge a real junction gap or invent a line from a
    component box.  The operation is a candidate-set cleanup, not a topology
    rule.
    """
    if not segments:
        return []
    axis_tolerance = max(1.5, min(3.5, 0.75 * max(1.0, float(stroke_width))))
    overlap_ratio_threshold = 0.82

    def interval(item: dict[str, Any]) -> tuple[str, float, float, float]:
        first, second = _segment_points(item)
        orientation = str(item.get("orientation") or "")
        if orientation == "horizontal":
            coordinate = (first[1] + second[1]) / 2.0
            start, end = sorted((first[0], second[0]))
        else:
            coordinate = (first[0] + second[0]) / 2.0
            start, end = sorted((first[1], second[1]))
        return orientation, coordinate, start, end

    def quality(item: dict[str, Any]) -> tuple[int, float, int]:
        source = str(item.get("source") or "")
        layers = item.get("local_fused_layers")
        layer_count = len(layers) if isinstance(layers, list) else 0
        evidence = 0
        if item.get("skeleton_path_evidence") or item.get("skeleton_component_evidence"):
            evidence += 3
        if item.get("local_red_green_evidence") or item.get("red_gap_track_evidence"):
            evidence += 3
        if "hough" in source:
            evidence += 2
        if item.get("split_from_observed_node"):
            evidence += 1
        try:
            first, second = _segment_points(item)
            length = math.hypot(second[0] - first[0], second[1] - first[1])
        except (KeyError, TypeError, ValueError):
            length = 0.0
        return evidence + layer_count, length, 1 if item.get("confidence") is not None else 0

    # Process the strongest/longest measured observation first.  This makes
    # the result stable while still retaining local red-terminal evidence when
    # it is the only observation of an interval.
    ordered = sorted(
        (dict(item) for item in segments),
        key=lambda item: quality(item),
        reverse=True,
    )
    output: list[dict[str, Any]] = []
    for candidate in ordered:
        try:
            orientation, coordinate, start, end = interval(candidate)
        except (KeyError, TypeError, ValueError):
            continue
        length = end - start
        if length <= 0.5:
            continue
        duplicate = False
        for existing in output:
            try:
                other_orientation, other_coordinate, other_start, other_end = interval(existing)
            except (KeyError, TypeError, ValueError):
                continue
            if orientation != other_orientation or abs(coordinate - other_coordinate) > axis_tolerance:
                continue
            overlap = min(end, other_end) - max(start, other_start)
            if overlap <= 0.5:
                # A true pair of adjacent graph edges may share one node;
                # never collapse it into one candidate.
                continue
            other_length = other_end - other_start
            if overlap / max(0.5, min(length, other_length)) < overlap_ratio_threshold:
                continue
            duplicate = True
            break
        if not duplicate:
            output.append(candidate)
    return output


def _snap_segments_to_local_terminal_seeds(
    segments: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    *,
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    stroke_width: float,
) -> tuple[list[dict[str, Any]], int]:
    """Snap existing wire endpoints to locally evidenced red/green terminals.

    This is deliberately an endpoint operation: it never creates a segment,
    bridges two global Hough lines, or adds red pixels to the wire mask.  A
    candidate endpoint must be close to a same-side seed, have the matching
    horizontal/vertical outward direction, and have green support along the
    short correction corridor.  Red is only a local transition witness.
    """
    if not segments or not seeds or green_mask.size == 0:
        return segments, 0
    height, width = green_mask.shape[:2]
    green = green_mask > 0
    red = (
        np.where(red_mask > 0, 255, 0).astype(np.uint8)
        if red_mask is not None
        else np.zeros_like(green, dtype=np.uint8)
    )
    directions = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    prepared: list[dict[str, Any]] = []
    for seed in seeds:
        point = seed.get("point_top_left")
        side = str(seed.get("side") or "").lower()
        if not isinstance(point, dict) or side not in directions:
            continue
        try:
            prepared.append({
                "x": float(point["x"]),
                "y": float(point["y"]),
                "side": side,
                "scale": float(seed.get("component_scale") or 32.0),
                "record": seed,
            })
        except (KeyError, TypeError, ValueError):
            continue
    output: list[dict[str, Any]] = []
    changed = 0
    for segment in segments:
        item = dict(segment)
        try:
            first, second = _segment_points(item)
        except (KeyError, TypeError, ValueError):
            output.append(item)
            continue
        orientation = str(item.get("orientation") or "")
        if orientation not in {"horizontal", "vertical"}:
            output.append(item)
            continue
        points = [first, second]
        for endpoint_index, endpoint in enumerate(points):
            best: tuple[float, dict[str, Any]] | None = None
            for seed in prepared:
                side = seed["side"]
                compatible = (
                    orientation == "horizontal" and side in {"left", "right"}
                ) or (
                    orientation == "vertical" and side in {"top", "bottom"}
                )
                if not compatible:
                    continue
                distance = math.hypot(endpoint[0] - seed["x"], endpoint[1] - seed["y"])
                radius = max(3.0, min(16.0, 0.22 * max(1.0, seed["scale"])))
                if distance > radius:
                    continue
                coordinate_error = (
                    abs(endpoint[1] - seed["y"])
                    if orientation == "horizontal"
                    else abs(endpoint[0] - seed["x"])
                )
                if coordinate_error > max(2.0, 1.5 * max(1.0, float(stroke_width))):
                    continue
                if best is None or distance < best[0]:
                    best = (distance, seed)
            if best is None:
                continue
            _distance, seed = best
            target = (seed["x"], seed["y"])
            samples = max(2, int(round(math.hypot(endpoint[0] - target[0], endpoint[1] - target[1]) * 2.0)))
            supported = 0
            red_supported = 0
            for sample_index in range(samples + 1):
                ratio = sample_index / float(samples)
                sx = int(round(endpoint[0] + ratio * (target[0] - endpoint[0])))
                sy = int(round(endpoint[1] + ratio * (target[1] - endpoint[1])))
                if not (0 <= sx < width and 0 <= sy < height):
                    continue
                y0, y1 = max(0, sy - 1), min(height, sy + 2)
                x0, x1 = max(0, sx - 1), min(width, sx + 2)
                supported += int(cv2.countNonZero(green_mask[y0:y1, x0:x1]) > 0)
                red_supported += int(cv2.countNonZero(red[y0:y1, x0:x1]) > 0)
            if supported / float(samples + 1) < 0.45:
                continue
            # A pure green terminal is valid.  When the correction crosses a
            # red-rendered transition, require at least one red observation.
            if red_supported == 0 and _distance > max(2.0, 1.5 * float(stroke_width)):
                continue
            points[endpoint_index] = target
            changed += 1
            item["local_terminal_snap_evidence"] = True
            item["local_terminal_seed"] = seed["record"]
        item["points_top_left"] = [_as_point(points[0]), _as_point(points[1])]
        output.append(item)
    return output, changed


def _pin_local_roi_ray_segments(
    image: np.ndarray,
    pin_anchor_by_node: dict[tuple[int, int], dict[str, Any]],
    component_boxes: list[tuple[float, float, float, float]],
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    *,
    stroke_width: float,
    skeleton_mask: np.ndarray | None = None,
    junction_records: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Trace a bounded pin-origin corridor through fused local ink.

    The corridor combines green wire pixels, red terminal pixels and
    low-saturation dark line pixels only inside the ROI of a validated pin.
    It is not a global red/black wire mask.  A ray is accepted only when it
    starts at a pin anchor, has continuous local support, and ends at a real
    gap or a direction change.  ROI reach and width are derived from the
    component scale, so this is usable across raster resolutions.
    """
    if not pin_anchor_by_node or image.ndim != 3:
        return []
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    dark = (gray < 220) & (saturation <= 100)
    green = green_mask > 0
    red = red_mask > 0 if red_mask is not None else np.zeros_like(green)
    # In the coloured renderer, achromatic pixels include pin numbers and
    # labels.  The local corridor is therefore driven by observed green wire
    # and red terminal layers; dark pixels must not form a free local bridge.
    fused = np.where(green | red, 255, 0).astype(np.uint8)
    skeleton = (
        skeleton_mask > 0
        if skeleton_mask is not None and skeleton_mask.shape == green_mask.shape
        else np.zeros_like(green)
    )
    skeleton_degree = _degree4_map(np.where(skeleton, 255, 0).astype(np.uint8))
    observed_junctions: list[tuple[float, float]] = []
    for record in junction_records or []:
        point = record.get("point_top_left") if isinstance(record, dict) else None
        if not isinstance(point, dict):
            continue
        try:
            observed_junctions.append((float(point["x"]), float(point["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    directions = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    output: list[dict[str, Any]] = []
    anchors = (
        pin_anchor_by_node.values()
        if isinstance(pin_anchor_by_node, dict)
        else (pin_anchor_by_node or [])
    )
    anchor_list = [item for item in anchors if isinstance(item, dict)]
    for anchor in anchors:
        point = anchor.get("point_top_left")
        side = str(anchor.get("side") or "").lower()
        if not isinstance(point, dict) or side not in directions:
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = directions[side]
        scale = 32.0
        index = anchor.get("component_index")
        try:
            if index is not None:
                box = component_boxes[int(index)]
                scale = min(box[2] - box[0], box[3] - box[1])
        except (TypeError, ValueError, IndexError):
            pass
        # Estimate the local pin pitch from validated anchors on the same
        # component side.  It bounds the ROI before a neighbouring pin or
        # annotation can be absorbed.  When a component has only one visible
        # anchor, the component scale remains the fallback; no case-specific
        # pixel span is used.
        point_axis = float(py if side in {"left", "right"} else px)
        same_side_axis: list[float] = []
        for other in anchor_list:
            if other is anchor:
                continue
            if other.get("component_key") != anchor.get("component_key"):
                continue
            if other.get("component_index") != anchor.get("component_index"):
                continue
            if str(other.get("side") or "").lower() != side:
                continue
            other_point = other.get("point_top_left")
            if not isinstance(other_point, dict):
                continue
            try:
                same_side_axis.append(
                    float(other_point["y"] if side in {"left", "right"} else other_point["x"])
                )
            except (KeyError, TypeError, ValueError):
                continue
        pitch = None
        if same_side_axis:
            gaps = sorted(
                abs(point_axis - value)
                for value in same_side_axis
                if abs(point_axis - value) > max(1.0, 2.0 * stroke_width)
            )
            if gaps:
                pitch = float(np.median(gaps))
        scale_bound = 0.9 * max(1.0, float(scale))
        pitch_bound = 2.5 * pitch if pitch is not None else scale_bound
        reach = max(
            6.0 * max(1.0, float(stroke_width)),
            min(128.0, scale_bound, pitch_bound),
        )
        local_span = min(max(1.0, float(scale)), pitch or max(1.0, float(scale)))
        lateral = max(
            2.0 * max(1.0, float(stroke_width)),
            min(8.0, 0.08 * local_span),
        )
        kernel_size = _odd_kernel_size(max(1.0, 0.75 * stroke_width), maximum=5)
        half = kernel_size // 2
        x0 = max(0, int(math.floor(min(px, px + dx * reach) - lateral - 2)))
        x1 = min(width - 1, int(math.ceil(max(px, px + dx * reach) + lateral + 2)))
        y0 = max(0, int(math.floor(min(py, py + dy * reach) - lateral - 2)))
        y1 = min(height - 1, int(math.ceil(max(py, py + dy * reach) + lateral + 2)))
        local = fused[y0:y1 + 1, x0:x1 + 1]
        local = cv2.morphologyEx(
            local,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
        )
        path: list[tuple[int, int, float]] = []
        previous_offset = 0.0
        missing = 0
        for step in range(int(math.ceil(reach)) + 1):
            gx = int(round(px + dx * step))
            gy = int(round(py + dy * step))
            candidates: list[tuple[float, float, int, int]] = []
            for offset in range(-int(math.ceil(lateral)), int(math.ceil(lateral)) + 1):
                xx = gx + int(round(-dy * offset))
                yy = gy + int(round(dx * offset))
                lx, ly = xx - x0, yy - y0
                if not (0 <= lx < local.shape[1] and 0 <= ly < local.shape[0]):
                    continue
                px0, px1 = max(0, lx - half), min(local.shape[1], lx + half + 1)
                py0, py1 = max(0, ly - half), min(local.shape[0], ly + half + 1)
                support = int(cv2.countNonZero(local[py0:py1, px0:px1]))
                if support <= 0:
                    continue
                score = float(support) - 0.75 * abs(float(offset) - previous_offset)
                candidates.append((score, float(offset), xx, yy))
            if not candidates:
                missing += 1
                if path and missing > 2:
                    break
                continue
            missing = 0
            _score, offset, xx, yy = max(candidates, key=lambda item: item[0])
            previous_offset = offset
            path.append((xx, yy, offset))
        if len(path) < 3:
            continue
        # A formal pin can be geometrically plausible while its green wire
        # starts a few pixels away (the intervening pixels are often the red
        # terminal stroke).  Do not let a red-only corridor become a wire:
        # the local ROI must contain a real green run and must touch the
        # global skeleton when one is available.  This is the gate that
        # permits unresolved/non-IC pins to participate in local recovery
        # without turning library coordinates into invented geometry.
        green_steps = 0
        red_steps = 0
        skeleton_contacts = 0
        contact_radius = max(1, min(4, int(math.ceil(1.5 * max(1.0, float(stroke_width))))))
        for sx, sy, _offset in path:
            gx0, gx1 = max(0, sx - contact_radius), min(width, sx + contact_radius + 1)
            gy0, gy1 = max(0, sy - contact_radius), min(height, sy + contact_radius + 1)
            green_steps += int(np.any(green[gy0:gy1, gx0:gx1]))
            red_steps += int(np.any(red[gy0:gy1, gx0:gx1]))
            if skeleton.size and np.any(skeleton[gy0:gy1, gx0:gx1]):
                skeleton_contacts += 1
        minimum_green_steps = max(2, int(math.ceil(1.0 * max(1.0, float(stroke_width)))))
        if green_steps < minimum_green_steps:
            continue
        if skeleton.size and skeleton_contacts == 0:
            # Very short real terminals can contain only one or two green
            # raster runs; the global skeleton intentionally rejects these
            # as isolated specks.  Keep such a path only when the bounded
            # local corridor has both a green observation and a red terminal
            # observation.  This is still image evidence at the pin origin,
            # not a library/order-based coordinate invention.
            if green_steps < minimum_green_steps or red_steps == 0:
                continue
        # Stop at the first stable direction change; the following branch is
        # handled by junction/path graph logic instead of being folded into a
        # single pin edge.
        endpoint_index = len(path) - 1
        for pos in range(3, len(path)):
            if abs(path[pos][2] - path[pos - 2][2]) > max(2.0, lateral * 0.6):
                endpoint_index = pos - 1
                break
        # A local colour corridor can remain straight through a junction. In
        # that case the offset test above cannot see the graph split. Prefer
        # the first observed skeleton endpoint/branch after leaving the pin.
        # The first two samples are ignored because the formal pin may project
        # onto a one-pixel skeleton run inside the terminal stroke.
        for pos in range(4, len(path)):
            sx, sy, _offset = path[pos]
            nearest = None
            nearest_distance = float("inf")
            search_radius = max(1, min(3, int(math.ceil(1.5 * max(1.0, stroke_width)))))
            for yy in range(max(0, sy - search_radius), min(height, sy + search_radius + 1)):
                for xx in range(max(0, sx - search_radius), min(width, sx + search_radius + 1)):
                    if not skeleton[yy, xx]:
                        continue
                    distance = math.hypot(float(xx - sx), float(yy - sy))
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest = (xx, yy)
            is_graph_node = nearest is not None and (
                int(skeleton_degree[nearest[1], nearest[0]]) >= 3
                or any(
                    math.hypot(float(nearest[0]) - jx, float(nearest[1]) - jy)
                    <= search_radius
                    for jx, jy in observed_junctions
                )
            )
            if is_graph_node:
                endpoint_index = pos
                break
        endpoint = path[endpoint_index]
        first = (px, py)
        second = (float(endpoint[0]), float(endpoint[1]))
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        if length < max(2.0, 1.25 * max(1.0, float(stroke_width))):
            continue
        if abs(second[1] - first[1]) <= max(3.0, lateral):
            coordinate = (first[1] + second[1]) / 2.0
            second = (second[0], coordinate)
            first = (first[0], coordinate)
            orientation = "horizontal"
        elif abs(second[0] - first[0]) <= max(3.0, lateral):
            coordinate = (first[0] + second[0]) / 2.0
            second = (coordinate, second[1])
            first = (coordinate, first[1])
            orientation = "vertical"
        else:
            orientation = "diagonal"
        output.append(
            {
                "points_top_left": [_as_point(first), _as_point(second)],
                "orientation": orientation,
                "source": "pin_local_fused_roi_ray",
                "confidence": 1.0,
                "pin_local_path_evidence": True,
                "local_fused_layers": ["green_skeleton", "red_terminal", "dark_line"],
                "endpoint_types": [
                    "component_terminal",
                    "local_junction_or_skeleton_node"
                    if endpoint_index < len(path) - 1
                    else "local_roi_stop",
                ],
                "terminal_bindings": [
                    {
                        key: anchor[key]
                        for key in (
                            "component_key",
                            "pin_key",
                            "component_index",
                            "side",
                            "formal_pin_roi_candidate",
                        )
                        if anchor.get(key) is not None
                    },
                    None,
                ],
            }
        )
    return output


def _local_terminal_color_segments(
    image: np.ndarray,
    terminal_seeds: list[dict[str, Any]],
    *,
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    skeleton_mask: np.ndarray | None = None,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Split the first few pixels after a validated terminal by observed ink.

    The green wire skeleton intentionally excludes the red symbol/terminal
    layer.  At a real component terminal the raster can consequently contain
    a short sequence such as ``red -> dark transition -> green``.  Hough is
    good at the long green part but cannot recover these sub-stroke endpoints.
    This helper emits only the bounded transition corridor starting at a
    validated terminal seed.  It never treats red as a global wire mask.

    Acceptance is evidence based:

    * the seed already comes from a validated pin or a red body side with a
      continuous green continuation;
    * the first corridor contains a green or red run and no unsupported gap;
    * dark pixels are retained only between a red and green run;
    * the green continuation is capped to the local transition, leaving the
      long run to the global skeleton/Hough stage.
    """
    if image.ndim != 3 or not terminal_seeds:
        return []
    if green_mask.shape[:2] != image.shape[:2]:
        return []
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    green = green_mask > 0
    red = red_mask > 0 if red_mask is not None else np.zeros_like(green)
    skeleton_support = (
        skeleton_mask > 0
        if skeleton_mask is not None and skeleton_mask.shape == green_mask.shape
        else np.zeros_like(green)
    )
    dark = (gray < 220) & (saturation <= 105)
    directions = {
        "left": (-1, 0),
        "right": (1, 0),
        "top": (0, -1),
        "bottom": (0, 1),
    }
    output: list[dict[str, Any]] = []
    for seed in terminal_seeds:
        point = seed.get("point_top_left")
        side = str(seed.get("side") or "").lower()
        if not isinstance(point, dict) or side not in directions:
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = directions[side]
        component_scale = float(seed.get("component_scale") or 32.0)
        # The local colour stage owns only the terminal transition.  The
        # bounded cap is tied to the measured component scale and stroke, so
        # it cannot consume a long wire that belongs to the global graph.
        # The colour stage owns only the terminal transition, but the red
        # rendered stub can be longer than the first green raster run.  Scale
        # the corridor from the measured stroke and component size instead
        # of using one fixed 5--8 px span.
        reach = int(
            max(
                3.0 * max(1.0, float(stroke_width)),
                min(16.0, 0.35 * max(1.0, component_scale)),
            )
        )
        lateral = int(max(1, min(3, round(0.045 * max(1.0, component_scale)))))
        samples: list[tuple[int, str, int]] = []
        for step in range(-1, reach + 1):
            gx = int(round(px + dx * step))
            gy = int(round(py + dy * step))
            scores = {"G": 0, "R": 0, "K": 0}
            for offset in range(-lateral, lateral + 1):
                xx = gx + int(round(-dy * offset))
                yy = gy + int(round(dx * offset))
                if not (0 <= xx < width and 0 <= yy < height):
                    continue
                if green[yy, xx]:
                    scores["G"] += 2
                if red[yy, xx]:
                    scores["R"] += 2
                if dark[yy, xx]:
                    scores["K"] += 1
            label = max(scores, key=scores.get)
            support = int(scores[label])
            samples.append((step, label if support >= 2 else ".", support))

        # A single antialiased hole is allowed only inside an already
        # supported transition.  A gap at the beginning/end remains a gap.
        for index in range(1, len(samples) - 1):
            if (
                samples[index][1] == "."
                and samples[index - 1][1] != "."
                and samples[index + 1][1] != "."
            ):
                samples[index] = (samples[index][0], samples[index - 1][1], samples[index][2])

        # Find the first supported corridor.  Do not follow a full long green
        # line here; that is the responsibility of the global wire backend.
        first = next((index for index, item in enumerate(samples) if item[1] != "."), None)
        if first is None:
            continue
        last_supported = first
        gap_count = 0
        for index in range(first + 1, len(samples)):
            if samples[index][1] == ".":
                gap_count += 1
                if gap_count > 1:
                    break
            else:
                gap_count = 0
                last_supported = index
        corridor = samples[first:last_supported + 1]
        if not any(item[1] in {"G", "R"} for item in corridor):
            continue
        # A local terminal must have a green continuation, unless the seed is
        # itself a validated pin whose direct red terminal is the only visible
        # part.  The latter is retained only for a short red run.
        has_green = any(item[1] == "G" for item in corridor)
        has_red = any(item[1] == "R" for item in corridor)
        if not has_green and not (has_red and len(corridor) <= 4):
            continue
        if skeleton_mask is not None:
            # A green pixel is useful for local fusion only when it reaches
            # the global 4-neighbour skeleton in the same bounded corridor.
            # This prevents an unconnected formal/library pin from turning a
            # nearby green annotation into a wire.
            skeleton_green = False
            support_radius = max(1, min(4, int(round(1.5 * float(stroke_width)))))
            for step, label, _support in corridor:
                if label != "G":
                    continue
                gx = int(round(px + dx * step))
                gy = int(round(py + dy * step))
                x0, x1 = max(0, gx - support_radius), min(width, gx + support_radius + 1)
                y0, y1 = max(0, gy - support_radius), min(height, gy + support_radius + 1)
                if np.any(skeleton_support[y0:y1, x0:x1]):
                    skeleton_green = True
                    break
            if not skeleton_green:
                continue

        # One validated seed represents one local terminal-to-wire corridor.
        # Export the complete measured corridor instead of one edge per colour
        # run; splitting red/green runs into 1-2 px fragments creates duplicate
        # protocol edges and loses the physical terminal semantics.
        start_step = corridor[0][0]
        end_step = corridor[-1][0]
        first_point = (px + dx * start_step, py + dy * start_step)
        second_point = (px + dx * end_step, py + dy * end_step)
        if start_step == end_step:
            second_point = (second_point[0] + dx, second_point[1] + dy)
        if math.hypot(second_point[0] - first_point[0], second_point[1] - first_point[1]) >= max(0.75, 0.75 * max(1.0, float(stroke_width))):
            output.append(
                {
                    "points_top_left": [_as_point(first_point), _as_point(second_point)],
                    "orientation": "horizontal" if dx else "vertical",
                    "source": "pin_local_color_transition",
                    "confidence": 1.0,
                    "pin_local_path_evidence": True,
                    "local_transition_label": "fused_terminal_wire_corridor",
                    "local_has_red_evidence": bool(has_red),
                    "local_has_green_evidence": bool(has_green),
                    "local_fused_layers": ["green_wire", "red_terminal", "dark_transition"],
                    "endpoint_types": ["validated_terminal", "local_color_transition"],
                    "terminal_bindings": [
                        {
                            key: seed[key]
                            for key in ("component_key", "pin_key", "component_index", "side")
                            if seed.get(key) is not None
                        },
                        None,
                    ],
                }
            )
    return output


def _filter_local_transitions_by_global_wire(
    candidates: list[dict[str, Any]],
    global_segments: list[dict[str, Any]],
    *,
    stroke_width: float,
    skeleton_mask: np.ndarray | None = None,
    junction_records: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Keep local colour transitions that join an outward global wire run.

    Red terminal pixels are useful only in the short transition corridor. A
    nearby red/green mark is not enough: the green end must agree with a
    global Hough path in orientation, lateral coordinate, and outward
    direction. This is a posterior evidence gate and never creates or moves
    a coordinate.
    """
    if not candidates:
        return []
    coordinate_tolerance = max(2.0, min(6.0, 2.5 * max(1.0, float(stroke_width))))
    gap_tolerance = max(3.0, min(10.0, 4.0 * max(1.0, float(stroke_width))))
    output: list[dict[str, Any]] = []
    degree = _degree4_map(skeleton_mask) if skeleton_mask is not None else None
    observed_junctions: list[tuple[float, float]] = []
    for record in junction_records or []:
        if not isinstance(record, dict):
            continue
        point = record.get("point_top_left")
        if not isinstance(point, dict):
            continue
        try:
            observed_junctions.append((float(point["x"]), float(point["y"])))
        except (KeyError, TypeError, ValueError):
            continue
    for candidate in candidates:
        points = candidate.get("points_top_left")
        orientation = candidate.get("orientation")
        if not isinstance(points, list) or len(points) != 2:
            continue
        if orientation not in {"horizontal", "vertical"}:
            continue
        try:
            terminal = _point(points[0])
            transition = _point(points[1])
        except (KeyError, TypeError, ValueError):
            continue
        if orientation == "horizontal":
            coordinate = transition[1]
            terminal_axis, transition_axis = terminal[0], transition[0]
        else:
            coordinate = transition[0]
            terminal_axis, transition_axis = terminal[1], transition[1]
        direction = 1.0 if transition_axis >= terminal_axis else -1.0
        attached = False
        for segment in global_segments:
            if segment.get("orientation") != orientation:
                continue
            try:
                first, second = _segment_points(segment)
            except (KeyError, TypeError, ValueError):
                continue
            segment_coordinate = (
                (first[1] + second[1]) / 2.0
                if orientation == "horizontal"
                else (first[0] + second[0]) / 2.0
            )
            if abs(segment_coordinate - coordinate) > coordinate_tolerance:
                continue
            low, high = sorted(
                (first[0], second[0]) if orientation == "horizontal" else (first[1], second[1])
            )
            if transition_axis < low - gap_tolerance or transition_axis > high + gap_tolerance:
                continue
            if direction > 0:
                outward_extension = high - transition_axis
            else:
                outward_extension = transition_axis - low
            if outward_extension < max(2.0, 1.5 * max(1.0, float(stroke_width))):
                continue
            attached = True
            break
        # A local terminal may be the only detected observation for a short
        # wire.  In that case allow it without a Hough parent only when its
        # green end reaches an observed skeleton node or junction.  A degree-2
        # interior pixel is deliberately insufficient: it would turn every
        # short red body/green annotation overlap into a wire.
        local_node = False
        if degree is not None and degree.size:
            tx, ty = int(round(transition[0])), int(round(transition[1]))
            # The local endpoint must be on the node itself (or one raster
            # pixel away), not merely in a broad neighbourhood where an
            # unrelated branch/text stroke can occur.
            radius = max(1, min(2, int(round(max(1.0, float(stroke_width))))))
            y0, y1 = max(0, ty - radius), min(degree.shape[0], ty + radius + 1)
            x0, x1 = max(0, tx - radius), min(degree.shape[1], tx + radius + 1)
            local_patch = degree[y0:y1, x0:x1]
            local_node = bool(np.any((local_patch == 1) | (local_patch >= 3)))
        if not local_node:
            local_node = any(
                math.hypot(transition[0] - x, transition[1] - y)
                <= max(2.0, 2.0 * max(1.0, float(stroke_width)))
                for x, y in observed_junctions
            )
        if attached or local_node:
            item = dict(candidate)
            item["global_wire_transition_evidence"] = True
            item["local_skeleton_node_evidence"] = bool(local_node)
            output.append(item)
    return output


def _local_terminal_raster_segments(
    image: np.ndarray,
    terminal_seeds: list[dict[str, Any]],
    *,
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    skeleton_mask: np.ndarray | None,
    global_segments: list[dict[str, Any]] | None,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Recover the short raster polyline at a coloured terminal.

    A rendered terminal is not always a single horizontal/vertical stroke.
    At a one-pixel anti-aliased offset it is commonly represented by
    ``red-horizontal -> one-pixel step -> green-horizontal``.  A straight
    local segment loses that topology and cannot match the real edge split.
    This helper derives a *bounded* three-part polyline from the observed
    red/green runs, then leaves the long green continuation to Hough/skeleton.

    The helper is intentionally local and evidence-gated.  A seed must have
    a same-side red/green transition (or a direct green terminal), and the
    green run must touch the global skeleton in the outward direction.  It
    never creates a path from a library pin coordinate alone.
    """
    if image.ndim != 3 or not terminal_seeds:
        return []
    if green_mask.shape[:2] != image.shape[:2]:
        return []
    height, width = image.shape[:2]
    green = green_mask > 0
    red = red_mask > 0 if red_mask is not None else np.zeros_like(green)
    skeleton = (
        skeleton_mask > 0
        if skeleton_mask is not None and skeleton_mask.shape == green_mask.shape
        else np.zeros_like(green)
    )
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    dark = (gray < 220) & (saturation <= 105)
    directions = {
        "left": (-1, 0),
        "right": (1, 0),
        "top": (0, -1),
        "bottom": (0, 1),
    }
    output: list[dict[str, Any]] = []
    stroke = max(1.0, float(stroke_width))
    lateral = max(1, min(3, int(math.ceil(1.25 * stroke))))
    skeleton_radius = max(1, min(4, int(math.ceil(1.5 * stroke))))

    def cross_section(
        x: int,
        y: int,
        dx: int,
        dy: int,
    ) -> tuple[int, int, int, int]:
        red_count = green_count = dark_count = skeleton_count = 0
        for offset in range(-lateral, lateral + 1):
            xx = x + int(round(-dy * offset))
            yy = y + int(round(dx * offset))
            if not (0 <= xx < width and 0 <= yy < height):
                continue
            red_count += int(red[yy, xx])
            green_count += int(green[yy, xx])
            dark_count += int(dark[yy, xx])
            y0, y1 = max(0, yy - skeleton_radius), min(height, yy + skeleton_radius + 1)
            x0, x1 = max(0, xx - skeleton_radius), min(width, xx + skeleton_radius + 1)
            skeleton_count += int(np.any(skeleton[y0:y1, x0:x1]))
        return red_count, green_count, dark_count, skeleton_count

    for seed in terminal_seeds:
        point = seed.get("point_top_left")
        side = str(seed.get("side") or "").lower()
        if not isinstance(point, dict) or side not in directions:
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = directions[side]
        # The terminal transition is a sub-stroke, not the complete wire.
        # The bound scales only with measured stroke width, so it is not tied
        # to a public case or an image's absolute resolution.
        max_forward = int(max(5, min(12, math.ceil(5.0 * stroke))))
        samples: list[tuple[int, int, int, int, int, int]] = []
        for step in range(-2, max_forward + 1):
            gx, gy = int(round(px + dx * step)), int(round(py + dy * step))
            red_count, green_count, dark_count, skeleton_count = cross_section(gx, gy, dx, dy)
            samples.append((step, red_count, green_count, dark_count, skeleton_count, gx))

        # Find a red run adjacent to the anchor.  Restricting the run to a
        # short band prevents the red IC body outline from becoming a path.
        red_steps = [item[0] for item in samples if item[1] > 0 and -2 <= item[0] <= 2]
        green_steps = [item[0] for item in samples if item[2] > 0 and item[0] >= -1]
        if not green_steps:
            continue
        first_green = min(green_steps)
        supported_green_steps = [
            item[0]
            for item in samples
            if item[0] >= first_green
            and item[2] > 0
            and item[4] > 0
        ]
        if not supported_green_steps:
            continue
        # A nearby green annotation or no-connect mark can touch the local
        # ROI without being a wire.  Require a same-side, axis-aligned green
        # run of measured-stroke length before emitting any raster piece.
        # One isolated antialias gap is allowed, but the run must continue
        # beyond the short transition into the global wire field.
        required_run = max(3, min(8, int(math.ceil(2.0 * stroke + 1.0))))
        green_run = 0
        green_gaps = 0
        for item in samples:
            step, _red_count, green_count, _dark_count, _skeleton_count, _gx = item
            if step < first_green:
                continue
            if green_count > 0:
                green_run += 1
                green_gaps = 0
            else:
                green_gaps += 1
                if green_gaps > 1:
                    break
            if green_run >= required_run:
                break
        if green_run < required_run:
            continue
        direct_pin_green_limit = max(1, int(math.ceil(1.5 * stroke)))
        if not red_steps and not (
            seed.get("pin_anchor_evidence") and first_green <= direct_pin_green_limit
        ):
            continue

        # The raster boundary is the last short red/transition pixel before
        # the green run.  One pixel of overlap is deliberate: it represents
        # the actual terminal-to-wire junction rather than a guessed gap.
        red_start = min(red_steps) if red_steps else max(-1, first_green - 1)
        red_end = max(red_steps) if red_steps else first_green - 1
        # The innermost red pixels can belong to the symbol body outline.
        # Keep one measured-stroke-length terminal span immediately before
        # the transition, rather than exporting the whole red body run.
        red_start = max(red_start, red_end - max(1, int(math.floor(stroke))))
        bridge_step = max(red_end + 1, first_green)
        green_end = min(
            first_green + max(1, int(math.ceil(2.5 * stroke))),
            max(supported_green_steps),
        )
        if green_end <= first_green:
            continue

        # Estimate centerline coordinates from the observed colour pixels.
        # ``ceil`` is stable for a two-row anti-aliased stroke and keeps a
        # real one-pixel raster step instead of collapsing it by rounding.
        def centerline(step: int, source: np.ndarray) -> tuple[float, float] | None:
            gx = int(round(px + dx * step))
            gy = int(round(py + dy * step))
            points: list[tuple[int, int]] = []
            for offset in range(-lateral, lateral + 1):
                xx = gx + int(round(-dy * offset))
                yy = gy + int(round(dx * offset))
                if 0 <= xx < width and 0 <= yy < height and source[yy, xx]:
                    points.append((xx, yy))
            if not points:
                return None
            if dx:
                return float(gx), float(math.ceil(float(np.mean([item[1] for item in points]))))
            return float(math.ceil(float(np.mean([item[0] for item in points])))), float(gy)

        red_line = centerline(red_end, red)
        green_line = centerline(first_green, green)
        if red_line is None:
            red_line = (px + dx * red_end, py + dy * red_end)
        if green_line is None:
            green_line = (px + dx * first_green, py + dy * first_green)

        # If the observed red and green centerlines are already collinear,
        # the global Hough candidate represents the same physical run.  Red
        # remains local terminal evidence, not an independent wire edge; the
        # raster supplement is reserved for an observed one-pixel step/bend.
        centerline_step = abs(red_line[1] - green_line[1]) if dx else abs(red_line[0] - green_line[0])
        if centerline_step < 0.5:
            continue

        # The local raster supplement must attach to a global wire candidate.
        # This rejects green symbol strokes and isolated terminal marks while
        # still allowing the short red/green transition that Hough itself
        # cannot represent.  The candidate is only used as corroborating
        # image geometry; its endpoints are not copied into the local path.
        if global_segments:
            expected_orientation = "horizontal" if dx else "vertical"
            continuation_point = (
                green_end,
                green_line[1] if dx else green_line[0],
            )
            attached = False
            for global_segment in global_segments:
                if global_segment.get("orientation") != expected_orientation:
                    continue
                try:
                    gp1, gp2 = _segment_points(global_segment)
                except (KeyError, TypeError, ValueError):
                    continue
                if expected_orientation == "horizontal":
                    coordinate = (gp1[1] + gp2[1]) / 2.0
                    start, end = sorted((gp1[0], gp2[0]))
                    forward_coordinate = px + dx * continuation_point[0]
                    coordinate_error = abs(coordinate - continuation_point[1])
                    directional_gap = (
                        max(0.0, start - forward_coordinate)
                        if dx > 0
                        else max(0.0, forward_coordinate - end)
                    )
                else:
                    coordinate = (gp1[0] + gp2[0]) / 2.0
                    start, end = sorted((gp1[1], gp2[1]))
                    forward_coordinate = py + dy * continuation_point[0]
                    coordinate_error = abs(coordinate - continuation_point[1])
                    directional_gap = (
                        max(0.0, start - forward_coordinate)
                        if dy > 0
                        else max(0.0, forward_coordinate - end)
                    )
                if coordinate_error <= max(2.0, 2.0 * stroke) and directional_gap <= max(3.0, 3.0 * stroke):
                    attached = True
                    break
            if not attached:
                continue
        red_start_point = (px + dx * red_start, py + dy * red_start)
        red_end_point = (px + dx * bridge_step, py + dy * bridge_step)
        green_start_point = (px + dx * bridge_step, py + dy * bridge_step)
        green_end_point = (px + dx * green_end, py + dy * green_end)
        if dx:
            red_start_point = (red_start_point[0], red_line[1])
            red_end_point = (red_end_point[0], red_line[1])
            green_start_point = (green_start_point[0], green_line[1])
            green_end_point = (green_end_point[0], green_line[1])
        else:
            red_start_point = (red_line[0], red_start_point[1])
            red_end_point = (red_line[0], red_end_point[1])
            green_start_point = (green_line[0], green_start_point[1])
            green_end_point = (green_line[0], green_end_point[1])

        pieces = [
            (red_start_point, red_end_point, "red_terminal"),
            (red_end_point, green_start_point, "terminal_step"),
            (green_start_point, green_end_point, "green_terminal_wire"),
        ]
        binding = {
            key: seed[key]
            for key in ("component_key", "pin_key", "component_index", "side")
            if seed.get(key) is not None
        }
        for first, second, label in pieces:
            if math.hypot(second[0] - first[0], second[1] - first[1]) < 0.75:
                continue
            output.append(
                {
                    "points_top_left": [_as_point(first), _as_point(second)],
                    "orientation": "horizontal" if abs(second[0] - first[0]) >= abs(second[1] - first[1]) else "vertical",
                    "source": "pin_local_raster_terminal_path",
                    "confidence": 1.0,
                    "pin_local_path_evidence": True,
                    "local_transition_label": label,
                    "local_has_red_evidence": bool(red_steps),
                    "local_has_green_evidence": True,
                    "endpoint_types": ["validated_terminal", "wire_terminal"],
                    "terminal_bindings": [binding, None],
                }
            )
    return output


def _local_pin_color_step_segments(
    image: np.ndarray,
    terminal_seeds: list[dict[str, Any]],
    *,
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    skeleton_mask: np.ndarray | None,
    global_segments: list[dict[str, Any]],
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Recover a validated pin's short red-to-green raster step.

    Some renderers draw the visible terminal as only one or two pixels before
    the green wire centerline.  A bbox-side scan cannot see that geometry when
    the public component box stops before the pin tip.  This helper starts at
    an already validated pin terminal and observes only a tiny outward color
    transition.  The longer path remains the responsibility of the global
    skeleton/Hough stages.
    """
    if image.ndim != 3 or red_mask is None or not terminal_seeds:
        return []
    if green_mask.shape != red_mask.shape or green_mask.shape != image.shape[:2]:
        return []
    green = green_mask > 0
    red = red_mask > 0
    height, width = green.shape[:2]
    stroke = max(1.0, float(stroke_width))
    lateral = max(1, min(3, int(round(stroke))))
    backtrack = max(1, min(3, int(math.ceil(1.5 * stroke))))
    forward = max(2, min(8, int(math.ceil(3.0 * stroke))))
    directions = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}

    def supported(source: np.ndarray, x: int, y: int) -> bool:
        x0, x1 = max(0, x - lateral), min(width, x + lateral + 1)
        y0, y1 = max(0, y - lateral), min(height, y + lateral + 1)
        return bool(np.any(source[y0:y1, x0:x1]))

    def attached(x: int, y: int, side: str) -> bool:
        radius = max(1, int(math.ceil(1.5 * stroke)))
        if skeleton_mask is not None and skeleton_mask.shape == green_mask.shape:
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            if np.any(skeleton_mask[y0:y1, x0:x1]):
                return True
        expected = "horizontal" if side in {"left", "right"} else "vertical"
        for segment in global_segments:
            if segment.get("orientation") != expected:
                continue
            try:
                first, second = _segment_points(segment)
            except (KeyError, TypeError, ValueError):
                continue
            if expected == "horizontal":
                coordinate = (first[1] + second[1]) / 2.0
                low, high = sorted((first[0], second[0]))
                if abs(coordinate - y) <= max(2.0, 2.0 * stroke) and low - radius <= x <= high + radius:
                    return True
            else:
                coordinate = (first[0] + second[0]) / 2.0
                low, high = sorted((first[1], second[1]))
                if abs(coordinate - x) <= max(2.0, 2.0 * stroke) and low - radius <= y <= high + radius:
                    return True
        return False

    output: list[dict[str, Any]] = []
    for seed in terminal_seeds:
        if not isinstance(seed, dict):
            continue
        point = seed.get("point_top_left")
        side = str(seed.get("side") or "").lower()
        if not isinstance(point, dict) or side not in directions:
            continue
        try:
            px, py = int(round(float(point["x"]))), int(round(float(point["y"])))
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = directions[side]
        labels: list[tuple[int, bool, bool]] = []
        for step in range(-backtrack, forward + 1):
            x, y = px + dx * step, py + dy * step
            if not (0 <= x < width and 0 <= y < height):
                labels.append((step, False, False))
                continue
            labels.append((step, supported(red, x, y), supported(green, x, y)))
        transitions: list[tuple[int, int]] = []
        for index, (step, has_red, _has_green) in enumerate(labels):
            if not has_red:
                continue
            for next_index in range(index + 1, min(len(labels), index + 3)):
                next_step, _next_red, next_green = labels[next_index]
                if next_green and next_step > step:
                    transitions.append((step, next_step))
                    break
        if not transitions:
            continue
        # Prefer the transition nearest the measured pin point.  Scanning
        # from the backward corridor would otherwise select the far edge of
        # a red body whenever several red pixels surround the seed.
        transition = min(
            transitions,
            key=lambda pair: (abs(pair[0]) + abs(pair[1]), abs(pair[1] - pair[0])),
        )
        first_step, second_step = transition
        first = (px + dx * first_step, py + dy * first_step)
        second = (px + dx * second_step, py + dy * second_step)
        if not attached(int(round(second[0])), int(round(second[1])), side):
            continue
        # A nearby skeleton pixel is not sufficient: the outward green run
        # must continue for at least a stroke-derived distance.  This removes
        # red text/body coincidences that happen to touch one green pixel,
        # while preserving genuine short pins on low-resolution sheets.
        continuation = 0
        for extra in range(0, forward + 1):
            xx = int(round(second[0] + dx * extra))
            yy = int(round(second[1] + dy * extra))
            if not (0 <= xx < width and 0 <= yy < height) or not supported(green, xx, yy):
                break
            continuation += 1
        if continuation < max(2, int(math.ceil(1.5 * stroke))):
            continue
        if math.hypot(second[0] - first[0], second[1] - first[1]) < 0.75:
            continue
        output.append(
            {
                "points_top_left": [_as_point(first), _as_point(second)],
                "orientation": "horizontal" if dx else "vertical",
                "source": "pin_local_raster_terminal_path",
                "confidence": 1.0,
                "local_raster_color_step_evidence": True,
                "local_fused_layers": ["red_terminal", "green_wire"],
                "endpoint_types": ["component_terminal", "wire_terminal"],
                "terminal_bindings": [
                    {
                        key: seed[key]
                        for key in ("component_key", "pin_key", "component_index", "side")
                        if seed.get(key) is not None
                    },
                    None,
                ],
            }
        )
        # Continue for one measured raster run on the green side.  This is a
        # local graph interval, not a long-wire replacement; the global Hough
        # path owns the remainder.
        green_start_step = None
        for step in range(second_step, forward + 1):
            xx = px + dx * step
            yy = py + dy * step
            if 0 <= xx < width and 0 <= yy < height and bool(green[yy, xx]):
                green_start_step = step
                break
        if green_start_step is not None:
            green_run = 0
            for step in range(
                green_start_step,
                min(forward, green_start_step + max(2, int(math.ceil(1.5 * stroke)))) + 1,
            ):
                xx = px + dx * step
                yy = py + dy * step
                if not (0 <= xx < width and 0 <= yy < height) or not bool(green[yy, xx]):
                    break
                green_run += 1
            if green_run >= max(2, int(math.ceil(1.5 * stroke))):
                green_first = (px + dx * green_start_step, py + dy * green_start_step)
                green_last_step = green_start_step + green_run - 1
                green_last = (px + dx * green_last_step, py + dy * green_last_step)
                if math.hypot(green_last[0] - green_first[0], green_last[1] - green_first[1]) >= 0.75:
                    output.append(
                        {
                            "points_top_left": [_as_point(green_first), _as_point(green_last)],
                            "orientation": "horizontal" if dx else "vertical",
                            "source": "pin_local_raster_terminal_path",
                            "confidence": 1.0,
                            "local_raster_green_continuation": True,
                            "local_fused_layers": ["green_wire"],
                            "endpoint_types": ["wire_terminal", "wire_node"],
                            "terminal_bindings": [
                                {
                                    key: seed[key]
                                    for key in ("component_key", "pin_key", "component_index", "side")
                                    if seed.get(key) is not None
                                },
                                None,
                            ],
                        }
                    )
    return output


def _red_component_boxes(
    image: np.ndarray,
    component_boxes: list[tuple[float, float, float, float]],
    *,
    stroke_width: float,
) -> list[tuple[float, float, float, float]]:
    """Find bounded red symbol bodies that are absent from detector boxes.

    This is only a local-terminal fallback.  It does not create components or
    keys.  Large page borders and red text are rejected; boxes overlapping an
    existing component box are not duplicated.
    """
    if image.ndim != 3:
        return []
    red = _red_terminal_mask(image)
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(red, 8)
    height, width = image.shape[:2]
    min_area = max(40, int(round(20.0 * max(1.0, float(stroke_width)) ** 2)))
    output: list[tuple[float, float, float, float]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = map(int, stats[label])
        if area < min_area:
            continue
        if box_width < max(12, int(round(6.0 * stroke_width))) or box_height < max(12, int(round(6.0 * stroke_width))):
            continue
        if box_width > 0.4 * width or box_height > 0.4 * height:
            continue
        candidate = (float(x), float(y), float(x + box_width - 1), float(y + box_height - 1))
        candidate_area = max(1.0, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
        duplicate = False
        for existing in component_boxes:
            ix = max(0.0, min(candidate[2], existing[2]) - max(candidate[0], existing[0]))
            iy = max(0.0, min(candidate[3], existing[3]) - max(candidate[1], existing[1]))
            intersection = ix * iy
            existing_area = max(1.0, (existing[2] - existing[0]) * (existing[3] - existing[1]))
            if intersection / min(candidate_area, existing_area) > 0.35:
                duplicate = True
                break
        if not duplicate:
            output.append(candidate)
    return output


def _red_box_terminal_seeds(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    green_mask: np.ndarray,
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Create local terminal seeds only for red bodies with green continuation.

    This is deliberately not a component detector and does not assign a key.
    It exists for the wire stage when a symbol bbox is absent from the object
    detector but its red body visibly meets a green wire.  A side is accepted
    only if a measured green run continues outward for several pixels; red
    text or an isolated red outline therefore cannot start a wire path.
    """
    if image.ndim != 3 or not boxes or green_mask.shape[:2] != image.shape[:2]:
        return []
    height, width = image.shape[:2]
    red_mask = _red_terminal_mask(image) > 0
    output: list[dict[str, Any]] = []
    for box_index, (x1, y1, x2, y2) in enumerate(boxes):
        ix0, ix1 = max(0, int(round(x1))), min(width, int(round(x2)) + 1)
        iy0, iy1 = max(0, int(round(y1))), min(height, int(round(y2)) + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        # The seed is for a locally observed red symbol/terminal body, not
        # for every detector rectangle.  This prevents a green wire merely
        # passing near an unrelated bbox from becoming a component terminal.
        red_body_pixels = int(red_mask[iy0:iy1, ix0:ix1].sum())
        if red_body_pixels < max(4, int(round(2.0 * max(1.0, float(stroke_width))))):
            continue
        scale = max(1.0, min(x2 - x1, y2 - y1))
        reach = int(max(5, min(16, round(0.22 * scale))))
        lateral = int(max(1, min(3, round(0.05 * scale))))
        box_seeds: list[dict[str, Any]] = []
        for side in ("left", "right", "top", "bottom"):
            if side in {"left", "right"}:
                axis_start = int(math.ceil(y1 + 2.0))
                axis_end = int(math.floor(y2 - 2.0))
                boundary = x1 if side == "left" else x2
                dx, dy = (-1, 0) if side == "left" else (1, 0)
            else:
                axis_start = int(math.ceil(x1 + 2.0))
                axis_end = int(math.floor(x2 - 2.0))
                boundary = y1 if side == "top" else y2
                dx, dy = (0, -1) if side == "top" else (0, 1)
            if axis_end < axis_start:
                continue
            supported_axes: list[int] = []
            for axis in range(axis_start, axis_end + 1):
                support_steps = 0
                consecutive = 0
                best_consecutive = 0
                for step in range(1, reach + 1):
                    gx = int(round(boundary + dx * step)) if dx else axis
                    gy = axis if dx else int(round(boundary + dy * step))
                    found = False
                    for offset in range(-lateral, lateral + 1):
                        xx = gx + int(round(-dy * offset))
                        yy = gy + int(round(dx * offset))
                        if 0 <= xx < width and 0 <= yy < height and green_mask[yy, xx]:
                            found = True
                            break
                    if found:
                        support_steps += 1
                        consecutive += 1
                        best_consecutive = max(best_consecutive, consecutive)
                    else:
                        consecutive = 0
                if support_steps >= max(3, int(round(0.45 * reach))) and best_consecutive >= 2:
                    supported_axes.append(axis)
            clusters: list[list[int]] = []
            for axis in supported_axes:
                if not clusters or axis - clusters[-1][-1] > max(2, int(round(1.5 * stroke_width))):
                    clusters.append([])
                clusters[-1].append(axis)
            for cluster in clusters:
                axis = float(np.median(cluster))
                point = (boundary, axis) if dx else (axis, boundary)
                box_seeds.append(
                    {
                        "point_top_left": _as_point(point),
                        "side": side,
                        "component_index": box_index,
                        "component_scale": scale,
                        "red_body_terminal_seed": True,
                    }
                )
        # A single green run beside a red glyph is not enough to identify a
        # missing component body.  Multiple independent lanes are the local
        # geometric evidence expected from a connector/IC-like red body.  A
        # one-pin part is still handled by the normal detector/pin path.
        if len(box_seeds) >= 2:
            output.extend(box_seeds)
    return output


def _red_box_terminal_segments(
    image: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Recover short line runs leaving locally observed red symbol bodies."""
    if image.ndim != 3 or not boxes:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    green = (hue >= 20) & (hue <= 110) & (saturation >= 15) & (value >= 40)
    red = ((hue <= 12) | (hue >= 168)) & (saturation >= 20) & (value >= 35)
    # This helper is also local to a detected red body; achromatic text must
    # not be allowed to create a terminal path.
    fused = np.where(green | red, 255, 0).astype(np.uint8)
    height, width = fused.shape[:2]
    output: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in boxes:
        scale = min(x2 - x1, y2 - y1)
        reach = int(round(max(8.0, min(64.0, 0.75 * max(1.0, scale)))))
        local_seed_records = _red_box_terminal_seeds(
            image,
            [(x1, y1, x2, y2)],
            np.where(green, 255, 0).astype(np.uint8),
            stroke_width=stroke_width,
        )
        seed_axes: dict[str, list[int]] = {"left": [], "right": [], "top": [], "bottom": []}
        for seed in local_seed_records:
            point = seed.get("point_top_left")
            side = str(seed.get("side") or "")
            if isinstance(point, dict) and side in seed_axes:
                axis_value = point.get("y") if side in {"left", "right"} else point.get("x")
                if axis_value is not None:
                    seed_axes[side].append(int(round(float(axis_value))))
        if not any(seed_axes.values()):
            continue
        for side in ("left", "right", "top", "bottom"):
            if side == "left":
                coordinate_values = seed_axes[side]
                dx, dy, boundary = -1, 0, x1
            elif side == "right":
                coordinate_values = seed_axes[side]
                dx, dy, boundary = 1, 0, x2
            elif side == "top":
                coordinate_values = seed_axes[side]
                dx, dy, boundary = 0, -1, y1
            else:
                coordinate_values = seed_axes[side]
                dx, dy, boundary = 0, 1, y2
            coordinate_values = sorted(
                {
                    value + delta
                    for value in coordinate_values
                    for delta in range(-2, 3)
                }
            )
            runs: list[tuple[int, int]] = []
            for coordinate in coordinate_values:
                best = 0
                missing = 0
                for step in range(1, reach + 1):
                    gx = int(round(boundary + dx * step)) if dx else int(coordinate)
                    gy = int(coordinate) if dx else int(round(boundary + dy * step))
                    if not (0 <= gx < width and 0 <= gy < height):
                        break
                    patch = fused[
                        max(0, gy - 1):min(height, gy + 2),
                        max(0, gx - 1):min(width, gx + 2),
                    ]
                    if cv2.countNonZero(patch):
                        best = step
                        missing = 0
                    else:
                        missing += 1
                        if missing > 2:
                            break
                if best >= max(2, int(round(1.25 * stroke_width))):
                    runs.append((int(coordinate), best))
            clusters: list[list[tuple[int, int]]] = []
            for item in runs:
                if not clusters or item[0] - clusters[-1][-1][0] > max(2, int(round(stroke_width))):
                    clusters.append([])
                clusters[-1].append(item)
            for cluster in clusters:
                length = max(item[1] for item in cluster)
                if length < max(2, int(round(1.25 * stroke_width))):
                    continue
                coordinate = float(np.median([item[0] for item in cluster]))
                if dx:
                    first = (float(boundary), coordinate)
                    second = (float(boundary + dx * length), coordinate)
                    orientation = "horizontal"
                else:
                    first = (coordinate, float(boundary))
                    second = (coordinate, float(boundary + dy * length))
                    orientation = "vertical"
                output.append(
                    {
                        "points_top_left": [_as_point(first), _as_point(second)],
                        "orientation": orientation,
                        "source": "red_body_local_terminal_path",
                        "confidence": 1.0,
                        "local_fused_layers": ["red_terminal", "green_skeleton", "dark_line"],
                        "endpoint_types": ["local_component_boundary", "local_terminal_stop"],
                    }
                )
    return output


def _component_red_green_transition_segments(
    image: np.ndarray,
    component_boxes: list[tuple[float, float, float, float]],
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    *,
    stroke_width: float,
    skeleton_mask: np.ndarray | None = None,
    global_segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Recover a measured red-terminal to green-wire transition per component.

    A detector box often covers the symbol body but not the short red pin
    stem.  In that rendering the green wire is tangent to the end of the red
    stem, so a normal ray from the box does not contain a long green run.  We
    therefore require two independent local observations:

    * a red run leaves one side of the detected body;
    * at its outward end, green pixels continue either along the normal or
      along the tangent direction, and are connected to the global green
      skeleton/Hough field.

    The returned segment is only the short observed transition.  It does not
    add red pixels to the global mask and it does not infer a pin from a box
    side without both colour observations.
    """
    if image.ndim != 3 or red_mask is None or not component_boxes:
        return []
    if green_mask.shape != red_mask.shape or green_mask.shape[:2] != image.shape[:2]:
        return []
    height, width = green_mask.shape[:2]
    green = green_mask > 0
    red = red_mask > 0
    stroke = max(1.0, float(stroke_width))
    output: list[dict[str, Any]] = []
    directions = {
        "left": (-1, 0),
        "right": (1, 0),
        "top": (0, -1),
        "bottom": (0, 1),
    }

    def patch_support(source: np.ndarray, x: int, y: int, radius: int) -> bool:
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        return bool(np.any(source[y0:y1, x0:x1]))

    def tangent_support(x: int, y: int, dx: int, dy: int, radius: int) -> int:
        # The tangent axis is perpendicular to the side normal.  Count the
        # longest contiguous green run through the transition pixel.
        best = 0
        for sign in (-1, 1):
            run = 0
            for step in range(0, max(4, int(math.ceil(5.0 * stroke))) + 1):
                xx = x + (-dy) * sign * step
                yy = y + dx * sign * step
                if patch_support(green, xx, yy, radius):
                    run += 1
                else:
                    break
            best = max(best, run)
        return best

    def global_attachment(x: int, y: int, side: str, radius: int) -> bool:
        if skeleton_mask is not None and skeleton_mask.shape == green_mask.shape:
            if patch_support(skeleton_mask > 0, x, y, max(radius, 1)):
                return True
        if not global_segments:
            return True
        expected = "horizontal" if side in {"left", "right"} else "vertical"
        for segment in global_segments:
            if segment.get("orientation") != expected:
                continue
            try:
                first, second = _segment_points(segment)
            except (KeyError, TypeError, ValueError):
                continue
            if expected == "horizontal":
                coordinate = (first[1] + second[1]) / 2.0
                start, end = sorted((first[0], second[0]))
                if abs(coordinate - y) <= max(2.0, 2.0 * stroke) and start - radius <= x <= end + radius:
                    return True
            else:
                coordinate = (first[0] + second[0]) / 2.0
                start, end = sorted((first[1], second[1]))
                if abs(coordinate - x) <= max(2.0, 2.0 * stroke) and start - radius <= y <= end + radius:
                    return True
        return False

    for component_index, box in enumerate(component_boxes):
        x1, y1, x2, y2 = box
        scale = max(1.0, min(x2 - x1, y2 - y1))
        # The search grows with the symbol scale and the measured stroke.  It
        # is capped relative to the box, so it cannot consume a distant label.
        reach = int(max(8.0, min(64.0, 0.9 * scale + 4.0 * stroke)))
        lateral = max(1, min(4, int(round(0.08 * scale + stroke))))
        support_radius = max(1, min(3, int(math.ceil(1.25 * stroke))))
        red_radius = max(0, min(1, int(math.floor(0.5 * stroke))))
        for side, (dx, dy) in directions.items():
            if side in {"left", "right"}:
                axis_start = int(math.floor(y1 - lateral))
                axis_end = int(math.ceil(y2 + lateral))
                boundary = x1 if side == "left" else x2
            else:
                axis_start = int(math.floor(x1 - lateral))
                axis_end = int(math.ceil(x2 + lateral))
                boundary = y1 if side == "top" else y2
            lane_candidates: list[tuple[int, int, int, int]] = []
            for axis in range(axis_start, axis_end + 1):
                red_steps: list[int] = []
                # Red must be present close to the body boundary and continue
                # outward; this excludes an unrelated red text glyph nearby.
                for step in range(0, reach + 1):
                    xx = int(round(boundary + dx * step)) if dx else axis
                    yy = axis if dx else int(round(boundary + dy * step))
                    if 0 <= xx < width and 0 <= yy < height and patch_support(red, xx, yy, red_radius):
                        red_steps.append(step)
                if len(red_steps) < max(2, int(math.ceil(stroke))):
                    continue
                # Use the red run that actually leaves the body boundary.
                # Taking ``max(red_steps)`` from a disconnected set can pick
                # up a nearby red label or a second symbol stroke and was the
                # main source of the candidate explosion on dense pages.
                near_boundary = min(red_steps)
                if near_boundary > max(2, int(math.ceil(1.5 * stroke))):
                    continue
                outer_red = near_boundary
                consecutive = 0
                last_supported = near_boundary
                for step in range(near_boundary, min(reach, near_boundary + max(4, int(math.ceil(8.0 * stroke)))) + 1):
                    xx = int(round(boundary + dx * step)) if dx else axis
                    yy = axis if dx else int(round(boundary + dy * step))
                    if 0 <= xx < width and 0 <= yy < height and patch_support(red, xx, yy, red_radius):
                        consecutive += 1
                        last_supported = step
                        outer_red = step
                    elif consecutive >= max(2, int(math.ceil(stroke))):
                        break
                    else:
                        consecutive = 0
                        outer_red = near_boundary
                if consecutive < max(2, int(math.ceil(stroke))):
                    continue
                # A one-stroke-thick symbol outline can satisfy the patch
                # test without being a terminal stem.  Require a normal red
                # run long enough to represent an actual rendered pin; the
                # scale is the measured stroke, not a case-specific pixel
                # constant.
                # Small symbols can render a genuine pin terminal as only a
                # few raster pixels.  Requiring four full stroke widths here
                # systematically deletes those real 1--2 px protocol edges.
                # Keep one stroke-scale red run as the minimum, while the
                # green continuation and global attachment checks below
                # remain mandatory.
                if outer_red - near_boundary + 1 < max(2, int(math.ceil(1.5 * stroke))):
                    continue
                # The green transition can be normal to the body or a tangent
                # wire crossing the red pin tip.  Search only after the red
                # run, with a stroke-derived gap allowance.
                first_green = None
                for gap in range(1, max(4, int(math.ceil(3.0 * stroke))) + 1):
                    step = outer_red + gap
                    xx = int(round(boundary + dx * step)) if dx else axis
                    yy = axis if dx else int(round(boundary + dy * step))
                    if 0 <= xx < width and 0 <= yy < height and bool(green[yy, xx]):
                        first_green = step
                        break
                if first_green is None:
                    continue
                gx = int(round(boundary + dx * first_green)) if dx else axis
                gy = axis if dx else int(round(boundary + dy * first_green))
                normal_run = 0
                for extra in range(0, max(4, int(math.ceil(4.0 * stroke))) + 1):
                    step = first_green + extra
                    xx = int(round(boundary + dx * step)) if dx else axis
                    yy = axis if dx else int(round(boundary + dy * step))
                    if 0 <= xx < width and 0 <= yy < height and bool(green[yy, xx]):
                        normal_run += 1
                    else:
                        break
                tangent_run = tangent_support(gx, gy, dx, dy, support_radius)
                if max(normal_run, tangent_run) < max(3, int(math.ceil(1.5 * stroke))):
                    continue
                if not global_attachment(gx, gy, side, support_radius):
                    continue
                lane_candidates.append((axis, outer_red, first_green, max(normal_run, tangent_run)))
            if not lane_candidates:
                continue
            # Adjacent raster lanes are one physical terminal.  Keep the lane
            # with the strongest green continuation, not every anti-aliased
            # row/column.
            clusters: list[list[tuple[int, int, int, int]]] = []
            for item in lane_candidates:
                if not clusters or item[0] - clusters[-1][-1][0] > max(2, int(math.ceil(1.5 * stroke))):
                    clusters.append([])
                clusters[-1].append(item)
            for cluster in clusters:
                axis, outer_red, first_green, _support = max(cluster, key=lambda item: item[3])
                red_point = (boundary + dx * outer_red, axis) if dx else (axis, boundary + dy * outer_red)
                green_point = (boundary + dx * first_green, axis) if dx else (axis, boundary + dy * first_green)
                if math.hypot(green_point[0] - red_point[0], green_point[1] - red_point[1]) < 0.75:
                    continue
                output.append(
                    {
                        "points_top_left": [_as_point(red_point), _as_point(green_point)],
                        "orientation": "horizontal" if dx else "vertical",
                        "source": "component_red_green_terminal_transition",
                        "confidence": 1.0,
                        "local_red_green_evidence": True,
                        "local_fused_layers": ["red_terminal", "green_wire", "junction_context"],
                        "endpoint_types": ["component_terminal", "wire_terminal"],
                        "terminal_bindings": [{"component_index": component_index, "side": side}, None],
                        "terminal_transition": "component_red_to_green",
                    }
                )
    return output


def _formal_pin_fused_transition_segments(
    image: np.ndarray,
    pin_anchors: list[dict[str, Any]],
    component_boxes: list[tuple[float, float, float, float]],
    green_mask: np.ndarray,
    red_mask: np.ndarray | None,
    *,
    skeleton_mask: np.ndarray,
    stroke_width: float,
    global_segments: list[dict[str, Any]] | None = None,
    junction_records: list[dict[str, Any]] | None = None,
    terminal_seeds: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Trace a formal-pin-local red/dark/green path with 4-neighbour BFS.

    The formal pin is only the search origin and direction prior.  Red pixels
    establish the local terminal transition; after the first green pixel is
    reached, traversal is restricted to the observed green/skeleton layer and
    continues to the first real skeleton endpoint or branch.  Thus the local
    result is a physical terminal-to-node path, not a red-to-first-green
    fragment and not a free extension from a bbox or library pin order.
    """
    extra_seeds = [item for item in (terminal_seeds or []) if isinstance(item, dict)]
    if image.ndim != 3 or not pin_anchors and not extra_seeds:
        return []
    if green_mask.shape != skeleton_mask.shape:
        return []
    height, width = green_mask.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    saturation = hsv[:, :, 1]
    green = green_mask > 0
    red = (
        red_mask > 0
        if red_mask is not None and red_mask.shape == green_mask.shape
        else np.zeros_like(green)
    )
    dark = (gray < 225) & (saturation <= 110)
    fused = green | red | dark
    graph_degree = _degree4_map(skeleton_mask)
    junction_points: list[tuple[float, float]] = []
    for record in junction_records or []:
        if record.get("classification") not in {
            "junction_candidate",
            "branch_candidate",
            "crossing_candidate",
        }:
            continue
        point = record.get("point_top_left")
        if isinstance(point, dict) and "x" in point and "y" in point:
            try:
                junction_points.append((float(point["x"]), float(point["y"])))
            except (TypeError, ValueError):
                continue
    stroke = max(1.0, float(stroke_width))
    directions = {"left": (-1, 0), "right": (1, 0), "top": (0, -1), "bottom": (0, 1)}
    output: list[dict[str, Any]] = []

    def near_green_global(point: tuple[int, int], side: str, radius: int) -> bool:
        x, y = point
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        if np.any(skeleton_mask[y0:y1, x0:x1]):
            return True
        if not global_segments:
            return True
        expected = "horizontal" if side in {"left", "right"} else "vertical"
        for segment in global_segments:
            if segment.get("orientation") != expected:
                continue
            try:
                first, second = _segment_points(segment)
            except (KeyError, TypeError, ValueError):
                continue
            if expected == "horizontal":
                coordinate = (first[1] + second[1]) / 2.0
                low, high = sorted((first[0], second[0]))
                if abs(coordinate - y) <= max(2.0, 2.0 * stroke) and low - radius <= x <= high + radius:
                    return True
            else:
                coordinate = (first[0] + second[0]) / 2.0
                low, high = sorted((first[1], second[1]))
                if abs(coordinate - x) <= max(2.0, 2.0 * stroke) and low - radius <= y <= high + radius:
                    return True
        return False

    def near_junction(point: tuple[int, int], radius: float) -> bool:
        return any(
            math.hypot(float(point[0]) - px, float(point[1]) - py) <= radius
            for px, py in junction_points
        )


    # Formal pin anchors remain the primary input.  Component-boundary seeds
    # are admitted only when the caller has already observed a real
    # skeleton-to-boundary intersection.  They carry no pin identity and can
    # therefore recover geometry for R/C/switch/LED without inventing a pin
    # number or a library-ordered terminal.
    origins = [*pin_anchors, *extra_seeds]
    for anchor in origins:
        if not isinstance(anchor, dict):
            continue
        point = anchor.get("point_top_left")
        side = str(anchor.get("side") or "").lower()
        index = anchor.get("component_index")
        if not isinstance(point, dict) or side not in directions or index is None:
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
            box = component_boxes[int(index)]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        dx, dy = directions[side]
        x1, y1, x2, y2 = box
        boundary = x1 if side == "left" else x2 if side == "right" else y1 if side == "top" else y2
        axis = py if side in {"left", "right"} else px
        scale = max(1.0, min(x2 - x1, y2 - y1))
        reach = int(max(10.0, min(96.0, 1.2 * scale + 5.0 * stroke)))
        # Keep the search lane tied to the local body scale, but narrower than
        # the full pin pitch.  A broad lane on ICs can enter the red body
        # outline and then falsely reach an unrelated green run.
        same_side_axes: list[float] = []
        for other in pin_anchors:
            if not isinstance(other, dict):
                continue
            if (
                other.get("component_index") != index
                or str(other.get("side") or "").lower() != side
            ):
                continue
            other_point = other.get("point_top_left")
            if not isinstance(other_point, dict):
                continue
            try:
                same_side_axes.append(
                    float(other_point["y"] if side in {"left", "right"} else other_point["x"])
                )
            except (KeyError, TypeError, ValueError):
                continue
        local_pitch = None
        if same_side_axes:
            gaps = [
                abs(axis - value)
                for value in same_side_axes
                if abs(axis - value) > max(1.5, 2.0 * stroke)
            ]
            if gaps:
                local_pitch = float(np.median(gaps))
        pitch_lane = 0.45 * local_pitch if local_pitch is not None else 0.16 * scale
        lateral = max(3, min(8, int(round(max(pitch_lane, 2.5 * stroke)))))
        normal_min = -max(2, int(math.ceil(1.5 * stroke)))
        normal_max = reach
        allowed: set[tuple[int, int]] = set()
        for yy in range(max(0, int(math.floor(y1 - reach))), min(height, int(math.ceil(y2 + reach)) + 1)):
            for xx in range(max(0, int(math.floor(x1 - reach))), min(width, int(math.ceil(x2 + reach)) + 1)):
                ox, oy = float(xx) - (boundary if dx else axis), float(yy) - (axis if dx else boundary)
                forward = ox * dx + oy * dy
                lateral_distance = abs(ox * dy - oy * dx) if dx or dy else float("inf")
                if normal_min <= forward <= normal_max and lateral_distance <= lateral:
                    allowed.add((xx, yy))
        if not allowed:
            continue
        # Start at red pixels close to the body side.  Formal coordinates are
        # not assumed accurate enough to be the BFS origin themselves.
        starts: list[tuple[int, int]] = []
        for xx, yy in allowed:
            ox, oy = float(xx) - (boundary if dx else axis), float(yy) - (axis if dx else boundary)
            forward = ox * dx + oy * dy
            if forward > max(3.0, 2.5 * stroke):
                continue
            if red[yy, xx]:
                starts.append((xx, yy))
        red_start_mode = bool(starts)
        if not starts:
            # Some renderers draw the terminal directly in green.  In that
            # case the red transition state is absent, but a green/skeleton
            # pixel immediately outside an already observed terminal is still
            # valid local evidence.  Do not search from the whole ROI: the
            # start must be within a stroke-derived radius and in the outward
            # corridor, otherwise a nearby wire/text stroke could be adopted.
            direct_radius = max(2.0, min(5.0, 2.5 * stroke))
            direct_candidates: list[tuple[float, tuple[int, int]]] = []
            for xx, yy in allowed:
                ox = float(xx) - (boundary if dx else axis)
                oy = float(yy) - (axis if dx else boundary)
                forward = ox * dx + oy * dy
                if forward < -max(1.0, stroke) or forward > direct_radius:
                    continue
                if not (green[yy, xx] or skeleton_mask[yy, xx]):
                    continue
                distance = math.hypot(float(xx) - px, float(yy) - py)
                if distance <= direct_radius:
                    direct_candidates.append((distance, (xx, yy)))
            if not direct_candidates:
                continue
            direct_candidates.sort(key=lambda item: item[0])
            starts = [direct_candidates[0][1]]
            red_start_mode = False
        queue = list(starts)
        parent: dict[tuple[int, int], tuple[int, int] | None] = {item: None for item in starts}
        target: tuple[int, int] | None = None
        green_seen: set[tuple[int, int]] = set()
        while queue and len(parent) <= 2048:
            current = queue.pop(0)
            cx, cy = current
            ox, oy = float(cx) - (boundary if dx else axis), float(cy) - (axis if dx else boundary)
            forward = ox * dx + oy * dy
            # Do not accept a green pixel still inside the body-side red
            # stroke; require a short outward transition first.  Once green
            # is reached, retain the state so the search can continue to a
            # true skeleton node instead of stopping at the colour boundary.
            if green[cy, cx] and forward >= max(2.0, 1.5 * stroke):
                green_seen.add(current)
            if green_seen and current in green_seen:
                degree = int(graph_degree[cy, cx])
                # Degree-zero/degree-one pixels are common at an antialiased
                # colour boundary and must not terminate the path merely
                # because the thinned raster is locally broken.  Prefer a
                # real branch; accept a degree-one endpoint only when there
                # is no outward skeleton continuation at that pixel.
                outward_neighbor = False
                for nx, ny in ((cx + dx, cy + dy),):
                    if 0 <= nx < width and 0 <= ny < height:
                        outward_neighbor = bool(green[ny, nx] or skeleton_mask[ny, nx])
                reached_node = (
                    degree >= 3
                    or (degree == 1 and not outward_neighbor)
                    or near_junction(current, max(1.5, 2.0 * stroke))
                ) and near_green_global(
                    current, side, max(1, int(math.ceil(1.5 * stroke)))
                )
                reached_roi_limit = forward >= normal_max - max(1.0, stroke)
                if reached_node or reached_roi_limit:
                    target = current
                    break
            for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                if (nx, ny) in parent or (nx, ny) not in allowed:
                    continue
                # Red/dark pixels are needed only to cross the terminal
                # transition.  After green contact, dark text and red body
                # outlines must not become alternative wire branches.
                if green_seen:
                    if not (green[ny, nx] or skeleton_mask[ny, nx]):
                        continue
                elif not (
                    (green[ny, nx] or red[ny, nx])
                    if anchor.get("red_green_local_evidence")
                    else fused[ny, nx]
                ):
                    continue
                parent[(nx, ny)] = current
                queue.append((nx, ny))
        if target is None:
            continue
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = target
        while current is not None:
            path.append(current)
            current = parent.get(current)
        path.reverse()
        green_index = next((pos for pos, item in enumerate(path) if green[item[1], item[0]]), None)
        if green_index is None:
            continue
        red_path = [item for item in path[:green_index] if red[item[1], item[0]]]
        if red_start_mode and len(red_path) < max(2, int(math.ceil(stroke))):
            continue
        # Skip the body-side part of a red outline, then retain the observed
        # red pin stem up to the green contact.  The skip distance scales with
        # the component and is capped by the measured stroke so a large IC
        # does not lose short pins.
        def forward_value(item: tuple[int, int]) -> float:
            return (item[0] - boundary) * dx + (item[1] - boundary) * dy if dx else (item[1] - boundary) * dy

        body_skip = max(2.0, min(10.0, 0.55 * scale))
        red_forward_values = [forward_value(item) for item in red_path]
        if red_start_mode and (
            not red_forward_values
            or max(red_forward_values) - min(red_forward_values) < max(2.0, 2.5 * stroke)
        ):
            continue
        if red_start_mode:
            red_start = next(
                (item for item in path[:green_index] if red[item[1], item[0]] and forward_value(item) >= body_skip),
                red_path[0],
            )
            transition_path = [red_start, *path[path.index(red_start) + 1:]]
        else:
            # Direct-green mode starts at the measured local terminal pixel;
            # no red/body segment is synthesized.  The path after the start
            # must remain green/skeleton-supported and ends at a real node.
            transition_path = path
        pieces = _path_to_segments(transition_path)
        for piece_index, (first, second) in enumerate(pieces):
            if math.hypot(second[0] - first[0], second[1] - first[1]) < 0.75:
                continue
            first_is_terminal = piece_index == 0
            second_is_junction = near_junction(
                (int(round(second[0])), int(round(second[1]))),
                max(1.5, 2.0 * stroke),
            )
            output.append(
                {
                    "points_top_left": [_as_point(first), _as_point(second)],
                    "orientation": "horizontal" if abs(second[1] - first[1]) <= abs(second[0] - first[0]) else "vertical",
                    "source": "pin_local_fused_bfs_transition",
                    "confidence": 1.0,
                    "local_red_green_evidence": bool(red_start_mode),
                    "local_green_terminal_evidence": not red_start_mode,
                    "local_fused_layers": [
                        *( ["red_terminal", "dark_transition"] if red_start_mode else [] ),
                        "green_skeleton",
                        "junction_context" if second_is_junction else "skeleton_node_context",
                    ],
                    "endpoint_types": [
                        "component_terminal" if first_is_terminal else "wire_node",
                        "wire_junction" if second_is_junction else "wire_node",
                    ],
                    "terminal_bindings": [
                        {
                            key: anchor[key]
                            for key in ("component_key", "pin_key", "component_index", "side")
                            if anchor.get(key) is not None
                        },
                        None,
                    ],
                }
            )
    return output


def _local_red_pin_segments(
    pin_anchor_by_node: dict[tuple[int, int], dict[str, Any]],
    red_mask: np.ndarray | None,
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Recover tiny red pin-terminal strokes in a bounded local corridor.

    Red ink is not a global wire layer.  Some renderers, however, draw the
    short terminal transition in red and the public edge protocol retains
    that visible geometry.  This helper considers only pixels close to an
    already validated pin anchor and never bridges two unrelated regions.
    """
    if not pin_anchor_by_node or red_mask is None or not cv2.countNonZero(red_mask):
        return []
    height, width = red_mask.shape[:2]
    radius = max(2.0, min(6.0, 2.5 * max(1.0, float(stroke_width))))
    forward_limit = max(2.0, min(6.0, 2.5 * max(1.0, float(stroke_width))))
    directions = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}
    output: list[dict[str, Any]] = []
    for anchor in pin_anchor_by_node.values():
        point = anchor.get("point_top_left")
        side = str(anchor.get("side") or "").lower()
        if not isinstance(point, dict) or side not in directions:
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = directions[side]
        candidates: list[tuple[float, float]] = []
        x0 = max(0, int(math.floor(px - forward_limit - radius - 1)))
        x1 = min(width - 1, int(math.ceil(px + forward_limit + radius + 1)))
        y0 = max(0, int(math.floor(py - forward_limit - radius - 1)))
        y1 = min(height - 1, int(math.ceil(py + forward_limit + radius + 1)))
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if not red_mask[y, x]:
                    continue
                ox, oy = float(x) - px, float(y) - py
                forward = ox * dx + oy * dy
                lateral = abs(ox * dy - oy * dx)
                if -forward_limit <= forward <= forward_limit and lateral <= radius:
                    signed_lateral = ox * dy - oy * dx
                    candidates.append((forward, signed_lateral))
        if len(candidates) < 2:
            continue
        low = min(item[0] for item in candidates)
        high = max(item[0] for item in candidates)
        if high - low < 0.75:
            continue
        center_lateral = float(np.median([item[1] for item in candidates]))
        # Use the median perpendicular coordinate so antialiasing on one side
        # of a thick stroke does not tilt the exported terminal segment.
        if abs(dx) > 0:
            y = py - center_lateral / dx
            first = (px + low * dx, y)
            second = (px + high * dx, y)
        else:
            x = px + center_lateral / dy
            first = (x, py + low * dy)
            second = (x, py + high * dy)
        output.append(
            {
                "points_top_left": [_as_point(first), _as_point(second)],
                "orientation": "horizontal" if abs(dx) > 0 else "vertical",
                "source": "pin_local_red_terminal",
                "confidence": 1.0,
                "pin_local_red_evidence": True,
                "endpoint_types": ["pin_terminal", "pin_terminal"],
                "terminal_bindings": [None, None],
                "red_support_pixel_count": len(candidates),
            }
        )
    return output


def _connected_red_terminal_segments(
    pin_anchor_by_node: dict[tuple[int, int], dict[str, Any]],
    red_mask: np.ndarray | None,
    green_mask: np.ndarray,
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Extract red terminal strokes only when locally connected to green.

    Red component ink is ambiguous globally.  The accepted red component must
    share a bounded 8-connected local component with the validated green
    skeleton node of the same pin.  This prevents red labels/body outlines
    near other pins from becoming wire edges.
    """
    if red_mask is None or not cv2.countNonZero(red_mask):
        return []
    if green_mask.shape != red_mask.shape:
        return []
    height, width = red_mask.shape[:2]
    radius = max(2, min(5, int(round(2.0 * max(1.0, float(stroke_width))))))
    forward_limit = max(6.0, min(16.0, 5.0 * max(1.0, float(stroke_width))))
    output: list[dict[str, Any]] = []
    directions = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}
    for node, anchor in pin_anchor_by_node.items():
        point = anchor.get("point_top_left")
        side = str(anchor.get("side") or "").lower()
        direction = directions.get(side)
        if not isinstance(point, dict) or direction is None:
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        dx, dy = direction
        x0 = max(0, int(math.floor(px - forward_limit * abs(dx) - radius - 2)))
        x1 = min(width - 1, int(math.ceil(px + forward_limit * abs(dx) + radius + 2)))
        y0 = max(0, int(math.floor(py - forward_limit * abs(dy) - radius - 2)))
        y1 = min(height - 1, int(math.ceil(py + forward_limit * abs(dy) + radius + 2)))
        local = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.uint8)
        red_local = red_mask[y0:y1 + 1, x0:x1 + 1] > 0
        green_local = green_mask[y0:y1 + 1, x0:x1 + 1] > 0
        for yy in range(local.shape[0]):
            for xx in range(local.shape[1]):
                gx, gy = x0 + xx, y0 + yy
                ox, oy = gx - px, gy - py
                forward = ox * dx + oy * dy
                lateral = abs(ox * dy - oy * dx)
                if -3.0 <= forward <= forward_limit and lateral <= radius:
                    local[yy, xx] = 255 if (red_local[yy, xx] or green_local[yy, xx]) else 0
        if not cv2.countNonZero(local):
            continue
        count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(local, 8)
        node_x = min(max(int(round(node[0])) - x0, 0), local.shape[1] - 1)
        node_y = min(max(int(round(node[1])) - y0, 0), local.shape[0] - 1)
        node_labels = {
            int(labels[yy, xx])
            for yy in range(max(0, node_y - 1), min(local.shape[0], node_y + 2))
            for xx in range(max(0, node_x - 1), min(local.shape[1], node_x + 2))
            if labels[yy, xx] > 0 and green_local[yy, xx]
        }
        if not node_labels:
            continue
        label = max(node_labels, key=lambda value: int(_stats[value, cv2.CC_STAT_AREA]))
        ys, xs = np.where((labels == label) & red_local)
        if len(xs) < 2:
            continue
        projections = [
            (float(x + x0) - px) * dx + (float(y + y0) - py) * dy
            for x, y in zip(xs, ys)
        ]
        if max(projections) - min(projections) < 0.75:
            continue
        # Keep only compact terminal strokes; long red paths belong to a
        # later topology-aware path stage rather than this pin-tip supplement.
        if max(projections) - min(projections) > max(6.0, 4.0 * max(1.0, float(stroke_width))):
            continue
        lateral_values = [
            (float(x + x0) - px) * dy - (float(y + y0) - py) * dx
            for x, y in zip(xs, ys)
        ]
        lateral = float(np.median(lateral_values))
        low, high = min(projections), max(projections)
        if abs(dx) > 0:
            line_y = py - lateral / dx
            first, second = (px + low * dx, line_y), (px + high * dx, line_y)
        else:
            line_x = px + lateral / dy
            first, second = (line_x, py + low * dy), (line_x, py + high * dy)
        output.append(
            {
                "points_top_left": [_as_point(first), _as_point(second)],
                "orientation": "horizontal" if abs(dx) else "vertical",
                "source": "pin_local_connected_red_terminal",
                "confidence": 1.0,
                "pin_local_path_evidence": True,
                "terminal_transition": "connected_red_green_component",
                "endpoint_types": ["component_terminal", "wire_terminal"],
                "terminal_bindings": [
                    {
                        key: anchor[key]
                        for key in ("component_key", "pin_key", "component_index", "side")
                        if anchor.get(key) is not None
                    },
                    None,
                ],
                "red_component_pixel_count": int(len(xs)),
            }
        )
    return output


def _pin_local_transition_segments(
    pin_anchor_by_node: dict[tuple[int, int], dict[str, Any]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Emit only the observed pin-to-wire transition path.

    A formal pin point and the first green/black skeleton pixel are often
    separated by the renderer's short coloured terminal or antialiasing
    band.  The long wire detector cannot represent that transition because
    it intentionally excludes red ink from the global wire mask.  When the
    pin anchor has already passed the local skeleton-connectivity test, the
    straight segment between the two observed points is a small, image-
    supported terminal transition; it is not a guessed wire extension.

    The length limit is derived from measured stroke width.  Isolated pins
    never enter ``pin_anchor_by_node`` and therefore cannot create output.
    """
    if not pin_anchor_by_node:
        return []
    maximum = max(3.0, min(12.0, 4.0 * max(1.0, float(stroke_width))))
    output: list[dict[str, Any]] = []
    for node, anchor in pin_anchor_by_node.items():
        point = anchor.get("point_top_left")
        if not isinstance(point, dict):
            continue
        try:
            first = (float(point["x"]), float(point["y"]))
            second = (float(node[0]), float(node[1]))
        except (KeyError, TypeError, ValueError):
            continue
        length = math.hypot(second[0] - first[0], second[1] - first[1])
        if not (0.75 < length <= maximum):
            continue
        dx, dy = second[0] - first[0], second[1] - first[1]
        orientation = "horizontal" if abs(dy) <= abs(dx) else "vertical" if abs(dx) < abs(dy) else "diagonal"
        output.append(
            {
                "points_top_left": [_as_point(first), _as_point(second)],
                "orientation": orientation,
                "source": "pin_terminal_local_transition",
                "confidence": 1.0,
                "endpoint_types": ["component_terminal", "wire_node"],
                "terminal_bindings": [
                    {
                        key: anchor[key]
                        for key in ("component_key", "pin_key", "component_index", "side")
                        if anchor.get(key) is not None
                    },
                    None,
                ],
                "pin_transition_evidence": True,
                "transition_length_px": round(float(length), 3),
            }
        )
    return output


def _validated_red_green_pin_transitions(
    pin_anchor_by_node: dict[tuple[int, int], dict[str, Any]],
    red_mask: np.ndarray | None,
    green_mask: np.ndarray,
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Return only short, observed red-to-green terminal transitions.

    This is the narrow local fusion permitted by the wire protocol.  A
    transition must connect a formal pin point to an already validated green
    skeleton node, contain both red and green pixels in the raster corridor,
    and remain shorter than a scale-derived terminal length.  Red pixels are
    never used to bridge two global wires or to create a free-standing edge.
    """
    if not pin_anchor_by_node or red_mask is None or not cv2.countNonZero(red_mask):
        return []
    if green_mask.shape != red_mask.shape:
        return []
    height, width = red_mask.shape[:2]
    maximum = max(2.0, min(12.0, 4.0 * max(1.0, float(stroke_width))))
    radius = max(1, min(3, int(round(1.25 * max(1.0, float(stroke_width))))))
    output: list[dict[str, Any]] = []
    for node, anchor in pin_anchor_by_node.items():
        point = anchor.get("point_top_left")
        if not isinstance(point, dict):
            continue
        try:
            px, py = float(point["x"]), float(point["y"])
        except (KeyError, TypeError, ValueError):
            continue
        nx, ny = float(node[0]), float(node[1])
        length = math.hypot(nx - px, ny - py)
        if not (0.75 < length <= maximum):
            continue
        samples = max(2, int(math.ceil(length * 2.0)))
        red_hits = 0
        green_hits = 0
        covered = 0
        for index in range(samples + 1):
            ratio = index / float(samples)
            x = int(round(px + ratio * (nx - px)))
            y = int(round(py + ratio * (ny - py)))
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            red_hit = cv2.countNonZero(red_mask[y0:y1, x0:x1]) > 0
            green_hit = cv2.countNonZero(green_mask[y0:y1, x0:x1]) > 0
            red_hits += int(red_hit)
            green_hits += int(green_hit)
            covered += int(red_hit or green_hit)
        if red_hits < 1 or green_hits < 1 or covered / float(samples + 1) < 0.6:
            continue
        output.append(
            {
                "points_top_left": [_as_point((px, py)), _as_point((nx, ny))],
                "orientation": (
                    "horizontal"
                    if abs(ny - py) <= 3.0
                    else "vertical"
                    if abs(nx - px) <= 3.0
                    else "diagonal"
                ),
                "source": "pin_local_red_green_transition",
                "confidence": 1.0,
                "pin_local_path_evidence": True,
                "terminal_transition": "red_to_validated_green",
                "endpoint_types": ["component_terminal", "wire_skeleton_node"],
                "terminal_bindings": [
                    {
                        key: anchor[key]
                        for key in ("component_key", "pin_key", "component_index", "side")
                        if anchor.get(key) is not None
                    },
                    None,
                ],
                "red_support_samples": red_hits,
                "green_support_samples": green_hits,
            }
        )
    return output


def _post_geometry_ocr_evidence(
    segments: list[dict[str, Any]],
    ocr_boxes: list[tuple[float, float, float, float]],
) -> list[dict[str, Any]]:
    """Attach OCR evidence after geometry; do not delete short wire paths."""
    output: list[dict[str, Any]] = []
    for segment in segments:
        item = dict(segment)
        overlap = _segment_ocr_overlap(segment, ocr_boxes)
        item["ocr_overlap"] = round(overlap, 4)
        # Require near-complete geometric overlap before OCR can reject a
        # raster wire candidate.  A real wire may pass beside/under a label,
        # so a loose overlap threshold would reduce recall.
        item["text_like"] = bool(overlap >= 0.9 and not any(
            endpoint == "component_terminal" for endpoint in item.get("endpoint_types", [])
        ))
        output.append(item)
    return output


def _remove_black_text_like_segments(
    segments: list[dict[str, Any]],
    *,
    colored_wire_layer: bool,
) -> list[dict[str, Any]]:
    """Apply OCR only as a posterior false-positive filter.

    In both renderers, a segment whose geometry is overwhelmingly inside an
    OCR box is rejected only after skeleton/path extraction and only when it
    has no observed terminal or junction endpoint.  OCR never participates in
    mask generation and never removes a terminal-connected short edge.
    """
    return [
        segment
        for segment in segments
        if not segment.get("text_like", False)
        or any(
            endpoint in {
                "component_terminal",
                "junction_node",
                "wire_junction",
            }
            for endpoint in segment.get("endpoint_types", [])
        )
    ]


def _remove_isolated_short_segments(
    segments: list[dict[str, Any]],
    *,
    min_length: float = 0.5,
) -> list[dict[str, Any]]:
    """Remove skeleton specks while preserving terminal-connected short wires."""
    output: list[dict[str, Any]] = []
    for segment in segments:
        p1, p2 = _segment_points(segment)
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        endpoint_types = segment.get("endpoint_types", [])
        if length <= 0.5:
            # Two identical points are not a valid geometric edge, even when
            # both observations came from a component terminal.
            continue
        if length < min_length and not endpoint_types:
            continue
        output.append(segment)
    return output


def _refine_segment_endpoints_from_raster(
    segments: list[dict[str, Any]],
    image: np.ndarray,
    wire_mask: np.ndarray,
    red_mask: np.ndarray | None,
    component_boxes: list[tuple[float, float, float, float]],
    ocr_boxes: list[tuple[float, float, float, float]],
    *,
    stroke_width: float,
) -> list[dict[str, Any]]:
    """Project existing axis runs to nearby raster-supported endpoints.

    Hough and skeleton endpoints often stop a few pixels inside an
    anti-aliased net-label or symbol terminal.  This helper only modifies an
    existing horizontal/vertical segment: it walks outward along its axis,
    requires local ink support, and stops at a known component/OCR boundary.
    It never creates a segment or bridges a blank corridor.
    """
    if not segments or image.ndim != 3 or wire_mask.size == 0:
        return segments
    height, width = wire_mask.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    green = wire_mask > 0
    red = red_mask > 0 if red_mask is not None else np.zeros_like(green)
    ink = (gray < 225) | ((saturation >= 20) & (value < 245))
    # The search range is tied to the measured raster stroke and capped to a
    # small local neighbourhood, so it cannot absorb a nearby text string.
    max_steps = max(2, min(6, int(round(2.0 * max(1.0, float(stroke_width)) + 1.0))))
    lateral_radius = max(1, min(3, int(round(max(1.0, float(stroke_width))))))

    def inside_box(x: float, y: float, boxes: list[tuple[float, float, float, float]]) -> bool:
        return any(x1 - 0.5 <= x <= x2 + 0.5 and y1 - 0.5 <= y <= y2 + 0.5 for x1, y1, x2, y2 in boxes)

    def near_ocr(x: float, y: float) -> bool:
        return inside_box(x, y, ocr_boxes)

    def supported(x: int, y: int, horizontal: bool) -> bool:
        if not (0 <= x < width and 0 <= y < height):
            return False
        for offset in range(-lateral_radius, lateral_radius + 1):
            xx = x
            yy = y
            if horizontal:
                yy += offset
            else:
                xx += offset
            if 0 <= xx < width and 0 <= yy < height:
                if green[yy, xx] or red[yy, xx] or ink[yy, xx]:
                    return True
        return False

    output: list[dict[str, Any]] = []
    for segment in segments:
        if segment.get("source") != "wire_hough_with_pin_local_path":
            output.append(segment)
            continue
        try:
            first, second = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            continue
        horizontal = abs(dx) >= abs(dy)
        if horizontal and abs(dy) > max(3.0, 2.0 * max(1.0, float(stroke_width))):
            output.append(segment)
            continue
        if not horizontal and abs(dx) > max(3.0, 2.0 * max(1.0, float(stroke_width))):
            output.append(segment)
            continue
        endpoints = [list(first), list(second)]
        for endpoint_index in (0, 1):
            point = endpoints[endpoint_index]
            direction = -1 if (point[0] < (first[0] + second[0]) / 2.0 if horizontal else point[1] < (first[1] + second[1]) / 2.0) else 1
            axis = point[0] if horizontal else point[1]
            fixed = point[1] if horizontal else point[0]
            extension = 0
            had_wire_support = False
            for step in range(1, max_steps + 1):
                axis_value = axis + direction * step
                x = int(round(axis_value)) if horizontal else int(round(fixed))
                y = int(round(fixed)) if horizontal else int(round(axis_value))
                if not (0 <= x < width and 0 <= y < height):
                    break
                # A component boundary is a valid terminal endpoint, but its
                # interior is never traversed.
                boundary_hit = inside_box(float(x), float(y), component_boxes)
                if near_ocr(float(x), float(y)) and not boundary_hit:
                    break
                has_support = supported(x, y, horizontal)
                if not has_support and not boundary_hit:
                    break
                if green[y, x] or red[y, x]:
                    had_wire_support = True
                extension = step
                if boundary_hit:
                    break
            if extension > 0 and had_wire_support:
                if horizontal:
                    point[0] = axis + direction * extension
                else:
                    point[1] = axis + direction * extension
        refined = dict(segment)
        refined["points_top_left"] = [_as_point(tuple(endpoints[0])), _as_point(tuple(endpoints[1]))]
        if refined["points_top_left"] != segment.get("points_top_left"):
            refined["endpoint_raster_refined"] = True
        output.append(refined)
    return output


def _remove_degenerate_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Final protocol guard against zero-length edges after path merging."""
    output: list[dict[str, Any]] = []
    for segment in segments:
        try:
            p1, p2 = _segment_points(segment)
        except (KeyError, TypeError, ValueError):
            continue
        if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) <= 0.5:
            continue
        output.append(segment)
    return output


def _merge_collinear_paths(
    segments: list[dict[str, Any]],
    junctions: list[dict[str, Any]],
    *,
    coordinate_tolerance: float = 2.0,
    gap_tolerance: float = 2.0,
) -> list[dict[str, Any]]:
    """Merge fragmented collinear skeleton paths without crossing junctions."""
    if not segments:
        return []
    junction_points = [
        _point(item["point_top_left"])
        for item in junctions
        if isinstance(item.get("point_top_left"), dict)
    ]
    # The tolerance is supplied in image pixels, but its useful scale is the
    # measured wire stroke rather than a case-specific constant.  Keep the
    # merge conservative: this only joins overlapping/touching collinear
    # pieces and never crosses a junction candidate.
    # The old implementation repeatedly scanned the whole list.  A dense
    # high-resolution schematic can produce hundreds of skeleton fragments,
    # making that O(n^2) loop look like a hang.  Grouping by orientation and
    # nearby axis coordinate gives the same conservative merge rule in
    # O(n log n), while still refusing to merge across a junction candidate.
    prepared: list[tuple[str, float, float, float, dict[str, Any]]] = []
    for segment in segments:
        orientation = segment.get("orientation")
        if orientation not in {"horizontal", "vertical"}:
            continue
        p1, p2 = _segment_points(segment)
        coordinate = (
            (p1[1] + p2[1]) / 2.0
            if orientation == "horizontal"
            else (p1[0] + p2[0]) / 2.0
        )
        start, end = sorted((p1[0], p2[0]) if orientation == "horizontal" else (p1[1], p2[1]))
        prepared.append((orientation, coordinate, start, end, dict(segment)))

    prepared.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    tracks: list[tuple[str, float, list[dict[str, Any]]]] = []
    for orientation, coordinate, _start, _end, segment in prepared:
        if tracks and tracks[-1][0] == orientation and abs(tracks[-1][1] - coordinate) <= coordinate_tolerance:
            old_orientation, old_coordinate, old_items = tracks[-1]
            new_coordinate = (old_coordinate * len(old_items) + coordinate) / (len(old_items) + 1)
            old_items.append(segment)
            tracks[-1] = (old_orientation, new_coordinate, old_items)
        else:
            tracks.append((orientation, coordinate, [segment]))

    result: list[dict[str, Any]] = []
    for orientation, _track_coordinate, items in tracks:
        horizontal = orientation == "horizontal"
        items.sort(
            key=lambda item: (
                sorted(
                    (
                        _segment_points(item)[0][0],
                        _segment_points(item)[1][0],
                    )
                    if horizontal
                    else (
                        _segment_points(item)[0][1],
                        _segment_points(item)[1][1],
                    )
                )[0],
                (
                    sum(point[1] for point in _segment_points(item)) / 2.0
                    if horizontal
                    else sum(point[0] for point in _segment_points(item)) / 2.0
                ),
            )
        )
        current: dict[str, Any] | None = None
        current_coordinate = 0.0
        current_start = 0.0
        current_end = 0.0

        def flush() -> None:
            nonlocal current
            if current is not None:
                result.append(current)
            current = None

        for item in items:
            p1, p2 = _segment_points(item)
            coordinate = (p1[1] + p2[1]) / 2.0 if horizontal else (p1[0] + p2[0]) / 2.0
            start, end = sorted((p1[0], p2[0]) if horizontal else (p1[1], p2[1]))
            if current is None:
                current = dict(item)
                current_coordinate, current_start, current_end = coordinate, start, end
                continue
            merged_start = min(current_start, start)
            merged_end = max(current_end, end)
            blocked = any(
                merged_start + 1.0 < (jx if horizontal else jy) < merged_end - 1.0
                and abs((jy if horizontal else jx) - current_coordinate) <= 3.0
                for jx, jy in junction_points
            )
            same_track = abs(coordinate - current_coordinate) <= coordinate_tolerance
            touching = start <= current_end + gap_tolerance
            if same_track and touching and not blocked:
                current["points_top_left"] = (
                    [_as_point((merged_start, (current_coordinate + coordinate) / 2.0)),
                     _as_point((merged_end, (current_coordinate + coordinate) / 2.0))]
                    if horizontal
                    else [_as_point(((current_coordinate + coordinate) / 2.0, merged_start)),
                          _as_point(((current_coordinate + coordinate) / 2.0, merged_end))]
                )
                current["confidence"] = max(
                    float(current.get("confidence", 0.0)), float(item.get("confidence", 0.0))
                )
                current["endpoint_types"] = list(dict.fromkeys(
                    list(current.get("endpoint_types", [])) + list(item.get("endpoint_types", []))
                ))
                current["merged_sources"] = list(current.get("merged_sources", [current.get("id")])) + list(
                    item.get("merged_sources", [item.get("id")])
                )
                current_coordinate = (current_coordinate + coordinate) / 2.0
                current_start, current_end = merged_start, merged_end
            else:
                flush()
                current = dict(item)
                current_coordinate, current_start, current_end = coordinate, start, end
        flush()

    for index, item in enumerate(result, start=1):
        item["id"] = f"wire_{index}"
    return result


def _compact_global_wire_fragments(
    segments: list[dict[str, Any]],
    *,
    protected_points: Iterable[tuple[float, float]] = (),
    stroke_width: float = 1.0,
    support_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Consolidate global wire fragments without crossing observed nodes.

    HoughLinesP frequently returns several adjacent intervals for one raster
    wire.  Emitting every interval as an independent protocol edge creates
    false positives even though the pixels are correct.  This helper is
    deliberately limited to global, axis-aligned candidates and uses only
    image-derived evidence:

    * intervals must be on the same measured axis track;
    * an actual positive gap must be supported by ``support_mask``;
    * an interval is never merged across a validated pin terminal or a
      junction point;
    * candidates carrying local red/green transition evidence are left
      untouched because those represent a separate terminal geometry layer.

    It is a topology-preserving compaction step, not a target-edge-count
    heuristic.  The protected points are the only places where a global wire
    is intentionally split.
    """
    if not segments:
        return []
    protected = [
        (float(x), float(y))
        for x, y in protected_points
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    axis_tolerance = max(1.0, min(2.5, 1.5 * max(1.0, float(stroke_width))))
    gap_tolerance = max(1.0, min(4.0, 2.5 * max(1.0, float(stroke_width))))

    def is_local(item: dict[str, Any]) -> bool:
        source = str(item.get("source") or "")
        # ``wire_hough_with_pin_local_path`` means a global Hough interval
        # was refined/merged with local evidence; it is still eligible for
        # global fragment compaction.  Only standalone local protocol edges
        # are protected here.
        return bool(
            item.get("local_red_green_evidence")
            or item.get("local_transition_edge_evidence")
            or source in {
                "wire_skeleton_local_graph_recall",
                "wire_skeleton_short_evidence",
                "pin_local_color_transition",
                "pin_local_raster_terminal_path",
                "pin_local_fused_bfs_transition",
                "pin_local_connected_red_terminal",
                "pin_local_red_green_transition",
            }
        )

    groups: dict[str, list[tuple[float, float, dict[str, Any]]]] = {
        "horizontal": [],
        "vertical": [],
    }
    passthrough: list[dict[str, Any]] = []
    for raw in segments:
        item = dict(raw)
        if is_local(item) or item.get("orientation") not in groups:
            passthrough.append(item)
            continue
        try:
            first, second = _segment_points(item)
        except (KeyError, TypeError, ValueError):
            passthrough.append(item)
            continue
        orientation = str(item["orientation"])
        if orientation == "horizontal":
            coordinate = (first[1] + second[1]) / 2.0
            start, end = sorted((first[0], second[0]))
        else:
            coordinate = (first[0] + second[0]) / 2.0
            start, end = sorted((first[1], second[1]))
        if end - start <= 0.5:
            passthrough.append(item)
            continue
        groups[orientation].append((coordinate, start, item))

    def gap_is_supported(
        orientation: str,
        coordinate: float,
        left_end: float,
        right_start: float,
    ) -> bool:
        if right_start <= left_end + 0.5:
            return True
        if right_start - left_end > gap_tolerance or support_mask is None:
            return False
        height, width = support_mask.shape[:2]
        left = int(math.floor(left_end))
        right = int(math.ceil(right_start))
        samples = []
        if orientation == "horizontal":
            y = int(round(coordinate))
            samples = [(x, y) for x in range(min(left, right), max(left, right) + 1)]
        else:
            x = int(round(coordinate))
            samples = [(x, y) for y in range(min(left, right), max(left, right) + 1)]
        if not samples:
            return False
        supported = 0
        for x, y in samples:
            if not (0 <= x < width and 0 <= y < height):
                continue
            x0, x1 = max(0, x - 1), min(width, x + 2)
            y0, y1 = max(0, y - 1), min(height, y + 2)
            supported += int(cv2.countNonZero(support_mask[y0:y1, x0:x1]) > 0)
        return supported / float(len(samples)) >= 0.5

    def blocked(
        orientation: str,
        coordinate: float,
        join_axis: float,
    ) -> bool:
        for px, py in protected:
            # ``join_axis`` is the longitudinal coordinate of the proposed
            # merge, while ``coordinate`` is the perpendicular track
            # coordinate.  Keep these two axes distinct: the previous
            # implementation compared x to y for horizontal tracks, so a
            # measured terminal transition could be compacted away.
            point_axis = px if orientation == "horizontal" else py
            point_coord = py if orientation == "horizontal" else px
            # Protect the actual graph join, not every point inside the
            # merged interval.  A raster skeleton node can be 1--2 pixels
            # inside a red-rendered terminal; the formal boundary point is
            # then near the interval endpoint, not at the Hough join.
            if (
                abs(point_axis - join_axis) <= max(0.75, float(stroke_width))
                and abs(point_coord - coordinate) <= axis_tolerance
            ):
                return True
        return False

    compacted: list[dict[str, Any]] = []
    for orientation in ("horizontal", "vertical"):
        values = groups[orientation]
        values.sort(key=lambda value: (value[0], _segment_points(value[2])[0][0 if orientation == "horizontal" else 1]))
        tracks: list[tuple[float, list[dict[str, Any]]]] = []
        for coordinate, _start, item in values:
            if tracks and abs(tracks[-1][0] - coordinate) <= axis_tolerance:
                old_coordinate, items = tracks[-1]
                items.append(item)
                tracks[-1] = ((old_coordinate * (len(items) - 1) + coordinate) / len(items), items)
            else:
                tracks.append((coordinate, [item]))
        for coordinate, items in tracks:
            decorated = []
            for item in items:
                first, second = _segment_points(item)
                if orientation == "horizontal":
                    start, end = sorted((first[0], second[0]))
                else:
                    start, end = sorted((first[1], second[1]))
                decorated.append((start, end, item))
            decorated.sort(key=lambda value: (value[0], value[1]))
            current: dict[str, Any] | None = None
            current_start = current_end = 0.0
            current_coordinate = coordinate

            def flush() -> None:
                nonlocal current
                if current is not None:
                    compacted.append(current)
                current = None

            for start, end, item in decorated:
                if current is None:
                    current = dict(item)
                    current_start, current_end = start, end
                    current_coordinate = coordinate
                    continue
                join_coordinate = (current_coordinate + coordinate) / 2.0
                can_join = (
                    start <= current_end + gap_tolerance
                    and gap_is_supported(orientation, join_coordinate, current_end, start)
                    and not blocked(orientation, join_coordinate, (current_end + start) / 2.0)
                )
                if not can_join:
                    flush()
                    current = dict(item)
                    current_start, current_end = start, end
                    current_coordinate = coordinate
                    continue
                merged_start, merged_end = min(current_start, start), max(current_end, end)
                if orientation == "horizontal":
                    current["points_top_left"] = [
                        _as_point((merged_start, join_coordinate)),
                        _as_point((merged_end, join_coordinate)),
                    ]
                else:
                    current["points_top_left"] = [
                        _as_point((join_coordinate, merged_start)),
                        _as_point((join_coordinate, merged_end)),
                    ]
                current["source"] = "wire_hough_with_pin_local_path"
                current["global_fragment_compacted"] = True
                current["merged_sources"] = list(current.get("merged_sources", [])) + [
                    item.get("source")
                ]
                current["confidence"] = max(
                    float(current.get("confidence", 0.0)),
                    float(item.get("confidence", 0.0)),
                )
                current_start, current_end = merged_start, merged_end
                current_coordinate = join_coordinate
            flush()
    return compacted + passthrough


def _remove_redundant_collinear_overlaps(
    segments: list[dict[str, Any]],
    *,
    stroke_width: float = 1.0,
) -> list[dict[str, Any]]:
    """Remove contained duplicate observations on one physical raster track.

    Hough and local path fusion can observe the same interval with different
    endpoints, e.g. a long parent and a shorter copy ending at the same pin.
    Such a copy does not add topology and is counted as a false edge.  This
    pass removes only a strictly contained, highly overlapping global copy;
    local transition edges and candidates with a distinct terminal identity
    remain untouched.  It never merges two separate tracks or uses a target
    edge count.
    """
    if len(segments) < 2:
        return segments
    coordinate_tolerance = max(1.0, min(2.0, 1.25 * max(1.0, float(stroke_width))))
    protected_sources = {
        "pin_local_color_transition",
        "pin_local_raster_terminal_path",
        "pin_local_fused_bfs_transition",
        "pin_local_connected_red_terminal",
        "pin_local_red_green_transition",
        "wire_skeleton_local_graph_recall",
        "wire_skeleton_short_evidence",
    }
    grouped: dict[tuple[str, int], list[tuple[float, float, int]]] = {}
    for index, item in enumerate(segments):
        orientation = str(item.get("orientation") or "")
        if orientation not in {"horizontal", "vertical"}:
            continue
        if str(item.get("source") or "") in protected_sources or item.get("local_transition_edge_evidence"):
            continue
        try:
            first, second = _segment_points(item)
        except (KeyError, TypeError, ValueError):
            continue
        coordinate = (first[1] + second[1]) / 2.0 if orientation == "horizontal" else (first[0] + second[0]) / 2.0
        start, end = sorted((first[0], second[0]) if orientation == "horizontal" else (first[1], second[1]))
        bucket = int(round(coordinate / coordinate_tolerance))
        grouped.setdefault((orientation, bucket), []).append((start, end, index))

    remove: set[int] = set()
    for values in grouped.values():
        # Start-first / end-descending makes a containing parent appear before
        # a contained fragment on the same measured track.
        values.sort(key=lambda value: (value[0], -value[1]))
        active: list[tuple[float, float, int]] = []
        for start, end, index in values:
            current = segments[index]
            current_length = end - start
            if current_length <= 1.0:
                continue
            for outer_start, outer_end, outer_index in reversed(active):
                if outer_index in remove:
                    continue
                if outer_start > start + 0.75 or outer_end < end - 0.75:
                    continue
                outer = segments[outer_index]
                overlap = max(0.0, min(outer_end, end) - max(outer_start, start))
                if overlap / max(1.0, current_length) < 0.8:
                    continue
                outer_bindings = {
                    (b.get("component_key"), b.get("pin_key"))
                    for b in outer.get("terminal_bindings", [])
                    if isinstance(b, dict) and b.get("pin_key")
                }
                current_bindings = {
                    (b.get("component_key"), b.get("pin_key"))
                    for b in current.get("terminal_bindings", [])
                    if isinstance(b, dict) and b.get("pin_key")
                }
                # If the shorter interval carries a different explicit pin,
                # it may be a genuine graph edge and must survive.
                if current_bindings - outer_bindings:
                    continue
                remove.add(index)
                break
            active.append((start, end, index))
    return [item for index, item in enumerate(segments) if index not in remove]


def _neighbors4(skeleton: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = point
    height, width = skeleton.shape[:2]
    result: list[tuple[int, int]] = []
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
            result.append((nx, ny))
    return result


def _neighbors8(skeleton: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
    x, y = point
    height, width = skeleton.shape[:2]
    result: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and skeleton[ny, nx]:
                result.append((nx, ny))
    return result


def _degree4_map(skeleton: np.ndarray) -> np.ndarray:
    binary = (skeleton > 0).astype(np.uint8)
    degree = np.zeros_like(binary, dtype=np.uint8)
    degree[1:, :] += binary[:-1, :]
    degree[:-1, :] += binary[1:, :]
    degree[:, 1:] += binary[:, :-1]
    degree[:, :-1] += binary[:, 1:]
    return degree


def _prune_skeleton_spurs(
    skeleton: np.ndarray,
    terminal_nodes: set[tuple[int, int]],
    stroke_width: float,
) -> np.ndarray:
    """Remove short non-terminal branches from a 4-connected skeleton."""
    work = np.where(skeleton > 0, 255, 0).astype(np.uint8)
    # Be conservative: a short branch that reaches a real skeleton branch is
    # kept for later junction classification.  It may be a genuine short
    # wire, while an isolated dead-end is the safer place to prune noise.
    max_spur = max(2, int(round(1.5 * stroke_width)))
    for _ in range(2):
        degrees = _degree4_map(work)
        endpoints = [
            (int(x), int(y))
            for y, x in zip(*np.where((work > 0) & (degrees == 1)))
        ]
        remove: set[tuple[int, int]] = set()
        for endpoint in endpoints:
            if any(math.hypot(endpoint[0] - node[0], endpoint[1] - node[1]) <= 2.0 for node in terminal_nodes):
                continue
            path = [endpoint]
            previous: tuple[int, int] | None = None
            current = endpoint
            reached_terminal = False
            reached_branch = False
            for _step in range(max_spur):
                next_points = [point for point in _neighbors4(work, current) if point != previous]
                if not next_points:
                    break
                if len(next_points) != 1:
                    reached_branch = True
                    break
                following = next_points[0]
                path.append(following)
                if any(math.hypot(following[0] - node[0], following[1] - node[1]) <= 2.0 for node in terminal_nodes):
                    reached_terminal = True
                    break
                previous, current = current, following
            if len(path) <= max_spur and not reached_terminal and not reached_branch:
                remove.update(path)
        if not remove:
            break
        for x, y in remove:
            work[y, x] = 0
    return work


def _repair_skeleton_endpoint_gaps(
    skeleton: np.ndarray,
    evidence_mask: np.ndarray,
    stroke_width: float,
) -> tuple[np.ndarray, int]:
    """Join only short, collinear skeleton endpoint gaps with image evidence.

    This is a local connectivity repair, not a global dilation.  Two endpoints
    are joined only when their outward directions face each other, their
    perpendicular offset is within the measured stroke scale, and the raw
    wire mask supports the missing interval.  Thus a short anti-aliased gap is
    repaired without bridging nearby parallel wires or crossing a junction.
    """
    work = np.where(skeleton > 0, 255, 0).astype(np.uint8)
    evidence = np.asarray(evidence_mask > 0, dtype=np.uint8)
    degrees = _degree4_map(work)
    endpoints = [
        (int(x), int(y))
        for y, x in zip(*np.where((work > 0) & (degrees == 1)))
    ]
    if len(endpoints) < 2:
        return work, 0
    max_gap = max(2, min(12, int(round(3.0 * stroke_width + 1.0))))
    lateral_tolerance = max(1, min(3, int(round(stroke_width))))
    candidates: list[tuple[float, tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
    for index, first in enumerate(endpoints):
        first_neighbors = _neighbors4(work, first)
        if len(first_neighbors) != 1:
            continue
        first_neighbor = first_neighbors[0]
        first_direction = (first[0] - first_neighbor[0], first[1] - first_neighbor[1])
        first_horizontal = abs(first_direction[0]) >= abs(first_direction[1])
        for second in endpoints[index + 1:]:
            second_neighbors = _neighbors4(work, second)
            if len(second_neighbors) != 1:
                continue
            second_neighbor = second_neighbors[0]
            second_direction = (second[0] - second_neighbor[0], second[1] - second_neighbor[1])
            second_horizontal = abs(second_direction[0]) >= abs(second_direction[1])
            if first_horizontal != second_horizontal:
                continue
            if first_horizontal:
                if abs(first[1] - second[1]) > lateral_tolerance:
                    continue
                if first_direction[0] * second_direction[0] >= 0:
                    continue
                distance = abs(first[0] - second[0])
                if distance <= 1 or distance - 1 > max_gap:
                    continue
                start, end = sorted((first[0], second[0]))
                line = [
                    (x, int(round((first[1] + second[1]) / 2.0)))
                    for x in range(start, end + 1)
                ]
            else:
                if abs(first[0] - second[0]) > lateral_tolerance:
                    continue
                if first_direction[1] * second_direction[1] >= 0:
                    continue
                distance = abs(first[1] - second[1])
                if distance <= 1 or distance - 1 > max_gap:
                    continue
                start, end = sorted((first[1], second[1]))
                line = [
                    (int(round((first[0] + second[0]) / 2.0)), y)
                    for y in range(start, end + 1)
                ]
            if not line:
                continue
            supported = sum(
                0 <= x < evidence.shape[1]
                and 0 <= y < evidence.shape[0]
                and bool(evidence[y, x])
                for x, y in line
            ) / float(len(line))
            if supported < 0.35:
                continue
            # A branch pixel in the interval means this is not a simple gap;
            # leave it for junction/crossing classification.
            if any(
                0 <= x < degrees.shape[1]
                and 0 <= y < degrees.shape[0]
                and degrees[y, x] >= 3
                for x, y in line[1:-1]
            ):
                continue
            score = float(distance) - 0.5 * supported
            candidates.append((score, first, second, line))

    repaired: set[tuple[int, int]] = set()
    repairs = 0
    for _score, first, second, line in sorted(candidates, key=lambda item: item[0]):
        if first in repaired or second in repaired:
            continue
        for x, y in line:
            work[y, x] = 255
        repaired.update((first, second))
        repairs += 1
    return work, repairs


def _filter_black_skeleton_components(
    skeleton: np.ndarray,
    terminal_nodes: set[tuple[int, int]],
    stroke_width: float,
) -> np.ndarray:
    """Reject isolated glyph-like components in the monochrome mode.

    Black schematics contain text and wires in the same intensity range.  A
    global skeleton therefore needs one structural filter, but the filter
    must not delete short real wires.  Components touching a detected
    component terminal are always preserved.  Other components are retained
    only when they have an elongated line shape or a real branch structure.
    All thresholds are derived from the measured stroke width.
    """
    if not cv2.countNonZero(skeleton):
        return skeleton
    binary = np.where(skeleton > 0, 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    degree = _degree4_map(skeleton)
    output = np.zeros_like(skeleton)
    terminal_radius = max(3.0, min(10.0, 3.0 * stroke_width))
    # Glyph fragments are commonly compact 5--12 px blobs, while a genuine
    # free wire has a longer axis.  Terminal-connected short wires bypass this
    # test; only isolated components use the stricter structural prior.
    long_size = max(12.0, 8.0 * stroke_width)
    thin_size = max(4.0, 3.0 * stroke_width)
    for label in range(1, count):
        x, y, width, height, area = map(int, stats[label])
        ys, xs = np.where(labels == label)
        if not len(xs):
            continue
        # Use the nearest skeleton pixel rather than the component centroid.
        # The previous diagonal/centroid tolerance could mark a nearby text
        # glyph as terminal-connected merely because the glyph's bbox was
        # large.  A real terminal connection has a short pixel distance.
        touches_terminal = False
        for tx, ty in terminal_nodes:
            if not (
                x - terminal_radius <= tx <= x + width + terminal_radius
                and y - terminal_radius <= ty <= y + height + terminal_radius
            ):
                continue
            distance_sq = np.min((xs.astype(np.float32) - float(tx)) ** 2 +
                                 (ys.astype(np.float32) - float(ty)) ** 2)
            if float(distance_sq) <= terminal_radius * terminal_radius:
                touches_terminal = True
                break
        branch_count = int(np.count_nonzero((labels == label) & (degree >= 3)))
        elongated = max(width, height) >= long_size and min(width, height) <= thin_size
        branched = (
            branch_count > 0
            and max(width, height) >= long_size
            and area >= max(8, int(round(3.0 * stroke_width)))
        )
        if touches_terminal or elongated or branched:
            output[labels == label] = 255
    return output


def _graph_nodes(skeleton: np.ndarray, terminal_nodes: set[tuple[int, int]]) -> set[tuple[int, int]]:
    binary = (skeleton > 0).astype(np.uint8)
    degrees = _degree4_map(skeleton)
    ys, xs = np.where((binary > 0) & ((degrees == 1) | (degrees >= 3)))
    nodes = {(int(x), int(y)) for x, y in zip(xs, ys)}
    nodes.update(terminal_nodes)
    return nodes


def _path_to_segments(path: list[tuple[int, int]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Simplify one skeleton path, preserving bends and short segments."""
    if len(path) < 2:
        return []
    contour = np.asarray(path, dtype=np.float32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(contour, epsilon=1.25, closed=False).reshape((-1, 2))
    if len(simplified) < 2:
        return []
    output: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for left, right in zip(simplified, simplified[1:]):
        x1, y1 = map(float, left)
        x2, y2 = map(float, right)
        if math.hypot(x2 - x1, y2 - y1) < 0.5:
            continue
        if abs(y2 - y1) <= 3.0:
            y = (y1 + y2) / 2.0
            y1 = y2 = y
        elif abs(x2 - x1) <= 3.0:
            x = (x1 + x2) / 2.0
            x1 = x2 = x
        output.append(((x1, y1), (x2, y2)))
    return output


def _trace_skeleton_segments(
    skeleton: np.ndarray,
    terminal_nodes: set[tuple[int, int]],
) -> list[dict[str, Any]]:
    # A thick raster junction can contain several adjacent degree>=3 pixels.
    # Treating every one as a separate graph node creates one-pixel fragments
    # and duplicate paths.  Collapse each 8-connected branch cluster to the
    # member closest to its centroid.  The cluster is still traversed using
    # the original 4-neighbour skeleton; only its graph endpoint is changed.
    binary = np.where(skeleton > 0, 255, 0).astype(np.uint8)
    degree = _degree4_map(skeleton)
    branch_mask = np.where((binary > 0) & (degree >= 3), 255, 0).astype(np.uint8)
    branch_labels_count, branch_labels, _stats, centroids = cv2.connectedComponentsWithStats(
        branch_mask, connectivity=8
    )
    branch_member_to_center: dict[tuple[int, int], tuple[int, int]] = {}
    branch_center_to_label: dict[tuple[int, int], int] = {}
    for label in range(1, branch_labels_count):
        ys, xs = np.where(branch_labels == label)
        if len(xs) == 0:
            continue
        cx, cy = centroids[label]
        center_index = int(np.argmin((xs.astype(np.float32) - float(cx)) ** 2 + (ys.astype(np.float32) - float(cy)) ** 2))
        center = (int(xs[center_index]), int(ys[center_index]))
        branch_center_to_label[center] = int(label)
        for x, y in zip(xs.tolist(), ys.tolist()):
            branch_member_to_center[(int(x), int(y))] = center

    def canonical(point: tuple[int, int]) -> tuple[int, int]:
        return branch_member_to_center.get(point, point)

    nodes = {
        canonical((int(x), int(y)))
        for y, x in zip(*np.where((binary > 0) & ((degree == 1) | (degree >= 3))))
    }
    nodes.update(canonical(point) for point in terminal_nodes if point in branch_member_to_center or skeleton[point[1], point[0]])

    def branch_exit_paths(center: tuple[int, int]) -> list[list[tuple[int, int]]]:
        """Return paths from a branch center to every adjacent non-branch pixel."""
        label = branch_center_to_label.get(center)
        if label is None:
            return [[center, neighbor] for neighbor in _neighbors4(skeleton, center)]
        queue = [center]
        parent: dict[tuple[int, int], tuple[int, int] | None] = {center: None}
        exits: list[tuple[tuple[int, int], tuple[int, int]]] = []
        while queue:
            current = queue.pop(0)
            for neighbor in _neighbors4(skeleton, current):
                if neighbor in parent:
                    continue
                if int(branch_labels[neighbor[1], neighbor[0]]) == label:
                    parent[neighbor] = current
                    queue.append(neighbor)
                else:
                    exits.append((current, neighbor))
        paths: list[list[tuple[int, int]]] = []
        for member, outside in exits:
            route = [member]
            while route[-1] != center:
                previous = parent.get(route[-1])
                if previous is None:
                    break
                route.append(previous)
            route.reverse()
            route.append(outside)
            if len(route) >= 2:
                paths.append(route)
        return paths

    visited: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    segments: list[dict[str, Any]] = []
    for start in sorted(nodes, key=lambda point: (point[1], point[0])):
        initial_paths = branch_exit_paths(start) if start in branch_center_to_label else [
            [start, neighbor] for neighbor in _neighbors4(skeleton, start)
        ]
        for initial in initial_paths:
            if len(initial) < 2:
                continue
            path = list(initial)
            for left, right in zip(path, path[1:]):
                edge = tuple(sorted((left, right)))
                if edge in visited:
                    path = []
                    break
                visited.add(edge)
            if not path:
                continue
            previous, current = path[-2], path[-1]
            while True:
                if current == start and len(path) > 2:
                    break
                if current in nodes and current != start:
                    break
                # If a normal path enters a branch cluster, finish at its
                # canonical center.  The cluster's internal pixels are not
                # exposed as separate protocol edges.
                if current in branch_member_to_center:
                    center = branch_member_to_center[current]
                    if center != current:
                        if path[-1] != center:
                            path.append(center)
                    break
                next_points = [point for point in _neighbors4(skeleton, current) if point != previous]
                if len(next_points) != 1:
                    break
                following = next_points[0]
                edge = tuple(sorted((current, following)))
                if edge in visited:
                    break
                visited.add(edge)
                if following in branch_member_to_center and branch_member_to_center[following] != following:
                    path.append(following)
                    path.append(branch_member_to_center[following])
                    break
                path.append(following)
                previous, current = current, following
            for first, second in _path_to_segments(path):
                dx, dy = second[0] - first[0], second[1] - first[1]
                orientation = "horizontal" if abs(dy) <= abs(dx) else "vertical" if abs(dx) < abs(dy) else "diagonal"
                segments.append(
                    {
                        "points_top_left": [
                            {"x": round(first[0], 2), "y": round(first[1], 2)},
                            {"x": round(second[0], 2), "y": round(second[1], 2)},
                        ],
                        "orientation": orientation,
                        "source": "wire_skeleton_path",
                        "confidence": 1.0,
                    }
                )
    # Keep exact-path endpoints distinct: unlike Hough merging, joining them
    # here could bridge a real junction or a component terminal.
    unique: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for segment in segments:
        p1, p2 = _segment_points(segment)
        key = tuple(round(value) for value in (*p1, *p2))
        reverse = tuple(round(value) for value in (*p2, *p1))
        unique[min(key, reverse)] = segment
    result = list(unique.values())
    for index, segment in enumerate(result, start=1):
        segment["id"] = f"wire_{index}"
    return result


def detect_wire_segments(
    image: np.ndarray,
    *,
    component_boxes_top_left: Iterable[list[float]] | None = None,
    ocr_items: Iterable[dict[str, Any]] | None = None,
    pin_anchors_top_left: Iterable[dict[str, Any]] | None = None,
    min_line_length: int = 8,
) -> list[dict[str, Any]]:
    """Return image-supported wire paths in top-left pixel coordinates."""
    geometry = detect_wire_geometry(
        image,
        component_boxes_top_left=component_boxes_top_left,
        ocr_items=ocr_items,
        pin_anchors_top_left=pin_anchors_top_left,
        min_line_length=min_line_length,
    )
    return geometry["segments_top_left"]


def detect_wire_geometry(
    image: np.ndarray,
    *,
    component_boxes_top_left: Iterable[list[float]] | None = None,
    ocr_items: Iterable[dict[str, Any]] | None = None,
    pin_anchors_top_left: Iterable[dict[str, Any]] | None = None,
    min_line_length: int = 8,
) -> dict[str, Any]:
    """Extract wire paths, terminals and junction candidates.

    Geometry is recovered before OCR filtering.  OCR is retained only as
    posterior evidence, because a text box can overlap a real short wire.
    The returned segments remain image-native top-left coordinates; protocol
    conversion is provided by :func:`segments_to_submission_edges`.
    """
    component_boxes = [_box(box) for box in (component_boxes_top_left or [])]
    component_boxes = [box for box in component_boxes if box is not None]
    pin_anchors = list(pin_anchors_top_left or [])
    ocr_boxes = _ocr_boxes(ocr_items)
    colored_wire_layer = _has_colored_wire_layer(image)
    mask = _wire_ink_mask(image, component_boxes=component_boxes)
    pin_source_mask = mask
    red_terminal_source = _red_terminal_mask(image) if colored_wire_layer else None
    if not colored_wire_layer:
        # Global black-line extraction is conservative to suppress text.  A
        # separate achromatic source is retained only for pin-local recovery
        # of short stubs and antialiased gaps.
        pin_source_mask = _achromatic_wire_source(image)
        _mask_component_interiors(pin_source_mask, component_boxes)
    high_resolution = max(image.shape[:2]) >= 2000
    # At 2K+ resolution the anti-aliased colour layer contains many isolated
    # one- and two-pixel specks.  They are removed before skeletonization;
    # connected short wire stubs remain intact.
    if colored_wire_layer and high_resolution:
        mask = _remove_tiny_ink_components(mask, min_area=3)
    stroke_width = _estimate_stroke_width(mask)
    # The green wire layer is already separated from red component outlines.
    # Masking its interior would cut real pin stubs inside a coarse body bbox.
    # Only the grayscale fallback needs the component-obstacle mask.
    if not colored_wire_layer:
        _mask_component_interiors(mask, component_boxes)
    # Do not remove long page-border-looking paths here.  The public targets
    # contain legitimate long net edges at the drawing boundary; whether a
    # path belongs to a net is decided later by topology, not by its length.
    # Closing repairs anti-alias gaps using the measured stroke width.  It is
    # deliberately bounded so neighbouring pins are not bridged.
    close_size = _odd_kernel_size(
        max(1.0, (1.25 if high_resolution else 0.6) * stroke_width),
        maximum=7,
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    skeleton = _thin(mask)
    terminals = _component_terminal_nodes(skeleton, component_boxes)
    pin_anchor_by_node = _pin_anchor_nodes(
        skeleton,
        pin_anchors,
        component_boxes,
        source_mask=pin_source_mask,
        red_mask=red_terminal_source,
        stroke_width=stroke_width,
    )
    recovery_source_mask = pin_source_mask
    if colored_wire_layer and red_terminal_source is not None:
        # Red is allowed only inside the validated pin-local recovery
        # corridor.  It is never added to the global wire mask or global
        # Hough input.
        recovery_source_mask = cv2.bitwise_or(pin_source_mask, red_terminal_source)
    skeleton, pin_anchor_by_node, pin_anchor_recovery_count = _recover_pin_anchor_paths(
        skeleton,
        pin_anchors,
        pin_anchor_by_node,
        component_boxes,
        recovery_source_mask,
        stroke_width,
    )
    terminals.update(pin_anchor_by_node)
    skeleton, skeleton_gap_repair_count = _repair_skeleton_endpoint_gaps(
        skeleton,
        mask,
        stroke_width,
    )
    if not colored_wire_layer:
        skeleton = _filter_black_skeleton_components(skeleton, terminals, stroke_width)
    # The green layer has already separated wire ink from text.  Pruning a
    # one- to two-stroke branch there can delete a real short terminal or a
    # junction arm before local ROI fusion sees it.  Keep the conservative
    # spur pruning for monochrome images, where glyph strokes are a genuine
    # ambiguity; coloured wires are filtered later by graph evidence.
    if not colored_wire_layer:
        skeleton = _prune_skeleton_spurs(skeleton, terminals, stroke_width)
    graph_skeleton = (
        _axis_supported_graph_skeleton(mask, stroke_width=stroke_width)
        if colored_wire_layer
        else skeleton
    )
    terminal_records = _terminal_records(
        terminals,
        component_boxes,
        pin_anchor_by_node=pin_anchor_by_node,
    )
    # In the green renderer, a coarse YOLO bbox side observation is not by
    # itself a graph node.  It remains available for posterior terminal
    # binding, but only pixel-supported branch nodes and validated pin
    # anchors are allowed to split the global wire path.  Otherwise a false
    # bbox-side hit fragments an otherwise continuous wire.  The monochrome
    # branch still uses component terminals to suppress text-connected paths.
    # Both validated pin anchors and component-boundary skeleton
    # intersections are real geometric terminals.  The latter are admitted
    # only after pixel-supported skeleton detection; a coarse bbox side alone
    # still never becomes a graph node.
    # A formal pin that passed the outward-path test is a graph node even
    # when it lies in the middle of a degree-2 raster run.  If it is added
    # only after tracing, the path has already been collapsed into a long
    # segment and the pin endpoint is lost.  Keep the key-free raster node
    # here; pin identity is attached later and never changes geometry.
    graph_terminals = set(terminals) | set(pin_anchor_by_node)
    # Keep the raw 4-neighbour skeleton for the global candidate graph.  The
    # axis-supported view is used for junction/topology validation and the
    # raw view is also required for short green terminal strokes; deleting
    # those pixels before local ROI fusion loses real wire endpoints.
    trace_skeleton = skeleton if colored_wire_layer else graph_skeleton
    skeleton_segments = _trace_skeleton_segments(trace_skeleton, graph_terminals)
    # A pin terminal can lie in the middle of a degree-2 skeleton run.  Split
    # only at anchors that already passed the local outward-path validation;
    # this recovers protocol edge endpoints without turning every bbox side
    # or library pin position into a graph node.
    pin_split_points = []
    for node in pin_anchor_by_node:
        # The raster node is guaranteed to lie on the skeleton.  The formal
        # boundary point can be a few pixels outside it and would otherwise
        # fail the split test before the later terminal projection step.
        pin_split_points.append((float(node[0]), float(node[1])))
    skeleton_segments = _split_axis_segments_at_points(
        skeleton_segments,
        pin_split_points,
        tolerance=max(2.0, min(6.0, 2.0 * max(1.0, float(stroke_width)))),
    )
    junction_source = graph_skeleton if colored_wire_layer else skeleton
    junctions = _junction_records(
        junction_source,
        terminals,
        stroke_width=stroke_width,
        wire_mask=mask,
        red_mask=red_terminal_source,
    )
    local_color_transition_count = 0
    pin_local_skeleton_candidate_count = 0
    fused_roi_candidate_count = 0
    local_terminal_snap_count = 0
    connected_red_candidate_count = 0
    validated_red_green_candidate_count = 0
    local_red_gap_merge_count = 0
    skeleton_recall_candidate_count = 0
    local_raster_terminal_candidate_count = 0
    fused_bfs_transition_count = 0
    fused_bfs_direct_green_count = 0
    fused_bfs_terminal_seed_count = 0
    component_red_green_seed_count = 0
    red_gap_track_union_count = 0
    red_terminal_overlap_union_count = 0
    red_terminal_parent_track_count = 0
    observed_node_parent_track_count = 0
    color_transition_split_points: list[tuple[float, float]] = []
    formal_pin_roi_candidate_count = 0
    formal_pin_roi_recovered_count = 0
    endpoint_dedup_tolerance = None
    axis_centerline_refinement_count = 0
    if colored_wire_layer:
        # The green renderer has a clean colour layer, but its 4-connected
        # skeleton can fragment long straight runs at antialiased pixels.
        # Use morphology/Hough only as a path candidate generator; every
        # candidate still comes from the same measured wire mask and is
        # therefore image-supported.  Black mode remains skeleton-first to
        # avoid turning text strokes into wires.
        hough_raw = _raw_segments(
            mask,
            component_boxes,
            ocr_boxes,
            # Keep short real wire intervals available to the terminal/local
            # fusion stage.  The scale is derived from the measured raster
            # stroke, not from a case-specific image size; longer paths are
            # still consolidated by the axis merge and skeleton evidence
            # gates below.
            min_line_length=max(3, int(math.ceil(2.5 * max(1.0, float(stroke_width))))),
            stroke_width=stroke_width,
        )
        # Global Hough is restricted to the actual green/black wire layer.
        # Red terminal pixels are consumed only by the bounded local ROI
        # searches below and never bridge unrelated global tracks.
        merge_support = mask
        hough_base = _merge_axis_segments(
            hough_raw,
            # Do not merge across a likely terminal/junction gap.  The
            # tolerance follows the measured stroke width and remains below
            # the scale at which two neighbouring schematic wires would be
            # bridged.  Local red/green fusion handles the short terminal
            # transition separately.
            gap_tolerance=max(2.0, min(5.0, 2.5 * max(1.0, float(stroke_width)))),
            support_mask=merge_support,
        )
        # Keep both observations for a pin terminal: the raster skeleton
        # point and the formal/boundary point can differ by a stroke or two.
        # Either one may be the closest evidence to a short red interruption.
        transition_anchor_points: list[tuple[float, float]] = []
        for record in terminal_records:
            if not isinstance(record, dict):
                continue
            for point_key in (
                "skeleton_point_top_left",
                "point_top_left",
                "component_boundary_reference_top_left",
            ):
                point = record.get(point_key)
                if not isinstance(point, dict) or "x" not in point or "y" not in point:
                    continue
                try:
                    transition_anchor_points.append((float(point["x"]), float(point["y"])))
                except (TypeError, ValueError):
                    continue
        hough_base, local_red_gap_merge_count = _merge_local_red_gap_segments(
            hough_base,
            mask,
            red_terminal_source,
            stroke_width=stroke_width,
            transition_anchor_points=transition_anchor_points,
        )
        red_gap_track_candidates = _red_gap_track_union_candidates(
            hough_base,
            mask,
            red_terminal_source,
            transition_anchor_points,
            stroke_width=stroke_width,
        )
        red_gap_track_union_count = len(red_gap_track_candidates)
        hough_base = _split_at_crossings(hough_base)
        hough_base, color_transition_split_points = _split_hough_at_local_color_transitions(
            image,
            hough_base,
            terminal_records,
            green_mask=mask,
            red_mask=red_terminal_source,
            stroke_width=stroke_width,
        )
        component_red_green_seeds = _component_red_green_terminal_seeds(
            image,
            component_boxes,
            green_mask=mask,
            red_mask=red_terminal_source,
            skeleton_mask=skeleton,
            global_segments=hough_base,
            stroke_width=stroke_width,
            existing_records=terminal_records,
        )
        component_red_green_seed_count = len(component_red_green_seeds)
        # These seeds are admitted as geometric terminals only after the
        # red-near-boundary/green-outward test above.  They carry no pin
        # identity, but their observed green point can split an already
        # detected Hough track at the physical terminal.
        # Keep these seeds in the local ROI fusion stage.  They are not yet
        # promoted to the global terminal list; promotion is performed only
        # by the image-supported local path itself, avoiding bbox-side seeds
        # splitting unrelated Hough tracks.
        hough_parent_source = list(hough_base)
        # Hough is intentionally allowed to recover a long continuous run,
        # but the contest graph is split at observed physical nodes.  Apply
        # this only to an already mask-supported Hough candidate: a pin or a
        # junction can partition a run, never create a free-standing wire.
        # This is the global counterpart of the local terminal ROI path and
        # keeps the terminal/junction topology visible to the later net stage.
        validated_terminal_records = [
            record
            for record in terminal_records
            if (
                record.get("pin_anchor_evidence", False)
                or (
                    isinstance(record.get("skeleton_point_top_left"), dict)
                    and record.get("side") in {"left", "right", "top", "bottom"}
                )
            )
        ]
        observed_node_parent_candidates = _observed_node_parent_track_candidates(
            hough_parent_source,
            mask,
            validated_terminal_records,
            junctions,
            stroke_width=stroke_width,
        )
        observed_node_parent_track_count = len(observed_node_parent_candidates)
        hough_base = _split_at_observed_nodes(
            hough_base,
            validated_terminal_records,
            [
                record
                for record in junctions
                if record.get("classification") == "junction_candidate"
            ],
            stroke_width=stroke_width,
        )
        red_terminal_parent_candidates = _red_terminal_parent_track_candidates(
            hough_parent_source,
            mask,
            red_terminal_source,
            terminal_records,
            stroke_width=stroke_width,
        )
        red_terminal_parent_track_count = len(red_terminal_parent_candidates)
        split_terminal_records = [
            {
                **anchor,
                "point_top_left": _as_point((float(node[0]), float(node[1]))),
                "pin_anchor_evidence": True,
            }
            for node, anchor in pin_anchor_by_node.items()
        ]
        split_junctions = [
            record for record in junctions
            if record.get("classification") == "junction_candidate"
        ]
        segments = [
            segment
            for segment in hough_base
            if _has_skeleton_support(segment, skeleton, stroke_width=stroke_width)
        ]
        # Preserve the measured global parent only for red-covered validated
        # terminals.  It is a recall supplement and does not replace the
        # graph-split child edges.
        segments.extend(observed_node_parent_candidates)
        segments.extend(red_terminal_parent_candidates)
        red_terminal_overlap_candidates = _red_gap_track_union_candidates(
            segments,
            mask,
            red_terminal_source,
            transition_anchor_points,
            stroke_width=stroke_width,
        )
        red_terminal_overlap_union_count = len(red_terminal_overlap_candidates)
        segments.extend(red_terminal_overlap_candidates)
        # The union candidate has independent green support on both sides
        # and red terminal evidence inside the gap.  Keep it separate from
        # the Hough parent so the real red-covered track is available without
        # erasing graph splits in the original pieces.
        segments.extend(red_gap_track_candidates)
        # Use skeleton intervals to refine the endpoints of an existing
        # Hough run when they are collinear and overlap/touch.  Unmatched
        # skeleton paths remain separate recall candidates below; this merge
        # only changes an already image-supported global line.
        segments = _merge_pin_local_paths(
            segments,
            skeleton_segments,
            stroke_width=stroke_width,
            append_unmatched=False,
        )
        short_candidates = _short_skeleton_wire_candidates(
            skeleton_segments,
            graph_skeleton,
            stroke_width=stroke_width,
        )
        known_segments = {
            tuple(round(value, 1) for point in item["points_top_left"] for value in (point["x"], point["y"]))
            for item in segments
        }
        for candidate in short_candidates:
            key = tuple(
                round(value, 1)
                for point in candidate["points_top_left"]
                for value in (point["x"], point["y"])
            )
            reverse_key = tuple(
                round(value, 1)
                for point in reversed(candidate["points_top_left"])
                for value in (point["x"], point["y"])
            )
            if key not in known_segments and reverse_key not in known_segments:
                segments.append(candidate)
                known_segments.add(key)
        # Local pin-origin paths recover short terminal-to-endpoint or
        # terminal-to-branch edges that the global Hough opening suppresses.
        # They are accepted only after the bounded 4-neighbour traversal has
        # reached a real skeleton endpoint/branch; arbitrary pin-to-pixel
        # extensions are not emitted.
        local_origin_by_node = dict(pin_anchor_by_node)
        for terminal in terminal_records:
            if terminal.get("pin_anchor_evidence", False):
                continue
            observed = terminal.get("skeleton_point_top_left") or terminal.get("point_top_left")
            if not isinstance(observed, dict):
                continue
            try:
                node_key = (int(round(float(observed["x"]))), int(round(float(observed["y"]))))
            except (KeyError, TypeError, ValueError):
                continue
            if terminal.get("side") in {"left", "right", "top", "bottom"}:
                local_origin_by_node.setdefault(node_key, dict(terminal))
        pin_local_candidates = _pin_local_skeleton_segments(
            skeleton,
            local_origin_by_node,
            stroke_width=stroke_width,
            component_boxes=component_boxes,
        )
        pin_local_skeleton_candidate_count = len(pin_local_candidates)
        segments = _merge_pin_local_paths(
            segments,
            pin_local_candidates,
            stroke_width=stroke_width,
        )
        if red_terminal_source is not None:
            connected_red_candidates = _connected_red_terminal_segments(
                pin_anchor_by_node,
                red_terminal_source,
                mask,
                stroke_width=stroke_width,
            )
            validated_red_green_candidates = _validated_red_green_pin_transitions(
                pin_anchor_by_node,
                red_terminal_source,
                mask,
                stroke_width=stroke_width,
            )
            connected_red_candidate_count = len(connected_red_candidates)
            validated_red_green_candidate_count = len(validated_red_green_candidates)
            # Keep these as local diagnostics for terminal validation.  Their
            # pixels can refine an existing path, but a short transition is
            # not emitted as a separate edge unless it survives the same
            # geometry candidate stage below.
        local_roi_anchors = dict(pin_anchor_by_node)
        # The connected-anchor map is intentionally strict for global graph
        # extraction.  For the local ROI stage we also inspect formal pin
        # coordinates that did not yet land on the green skeleton.  They are
        # not promoted to global terminals and cannot create a wire by
        # themselves: ``_pin_local_roi_ray_segments`` requires a measured
        # green run plus contact with the global skeleton.  This separates
        # "start the local search at the real pin tip" from "accept a wire".
        formal_pin_roi_anchor_keys = {
            (
                anchor.get("component_key"),
                anchor.get("pin_key"),
            )
            for anchor in local_roi_anchors.values()
            if isinstance(anchor, dict)
        }
        for formal_anchor in pin_anchors:
            if not isinstance(formal_anchor, dict):
                continue
            point = formal_anchor.get("point_top_left")
            side = str(formal_anchor.get("side") or "").lower()
            if (
                not isinstance(point, dict)
                or side not in {"left", "right", "top", "bottom"}
                or "x" not in point
                or "y" not in point
            ):
                continue
            identity = (formal_anchor.get("component_key"), formal_anchor.get("pin_key"))
            if identity in formal_pin_roi_anchor_keys:
                continue
            try:
                node_key = (int(round(float(point["x"]))), int(round(float(point["y"]))))
            except (KeyError, TypeError, ValueError):
                continue
            candidate = dict(formal_anchor)
            component_index = candidate.get("component_index")
            scale = 32.0
            try:
                if component_index is not None:
                    box = component_boxes[int(component_index)]
                    scale = min(box[2] - box[0], box[3] - box[1])
            except (TypeError, ValueError, IndexError):
                pass
            candidate["component_scale"] = round(float(scale), 3)
            candidate["formal_pin_roi_candidate"] = True
            local_roi_anchors.setdefault(node_key, candidate)
            formal_pin_roi_anchor_keys.add(identity)
            formal_pin_roi_candidate_count += 1
        # Generic component terminals are accepted here only when they were
        # found as an actual skeleton-to-component intersection.  This makes
        # the same local terminal/green-wire fusion available to R/C/switch/
        # LED symbols without treating a coarse YOLO bbox side as a pin.
        for terminal in terminal_records:
            if terminal.get("pin_anchor_evidence", False):
                continue
            if terminal.get("source") != "component_boundary_skeleton_intersection":
                continue
            point = terminal.get("point_top_left")
            side = str(terminal.get("side") or "").lower()
            if not isinstance(point, dict) or side not in {"left", "right", "top", "bottom"}:
                continue
            try:
                node = terminal.get("skeleton_point_top_left") or point
                node_key = (int(round(float(node["x"]))), int(round(float(node["y"]))))
            except (KeyError, TypeError, ValueError):
                continue
            local_roi_anchors.setdefault(node_key, dict(terminal))
        fused_roi_candidates = _pin_local_roi_ray_segments(
            image,
            local_roi_anchors,
            component_boxes,
            mask,
            red_terminal_source,
            stroke_width=stroke_width,
            skeleton_mask=skeleton,
            junction_records=junctions,
        )
        fused_roi_candidate_count = len(fused_roi_candidates)
        formal_pin_roi_recovered_count = sum(
            1 for candidate in fused_roi_candidates
            if candidate.get("terminal_bindings")
            and isinstance(candidate["terminal_bindings"][0], dict)
            and candidate["terminal_bindings"][0].get("formal_pin_roi_candidate")
        )
        segments = _merge_pin_local_paths(
            segments,
            fused_roi_candidates,
            stroke_width=stroke_width,
        )
        # Preserve the first colour-transition pieces as independent edges.
        # They must not be unioned with the long Hough run: the contest
        # geometry has a real terminal/transition endpoint at this boundary.
        transition_seeds = [dict(anchor) for anchor in pin_anchor_by_node.values()]
        # Generic component terminals are also valid geometric origins.  They
        # come from an observed skeleton-to-component boundary intersection,
        # not from a library pin order or an unverified bbox side.
        for terminal in terminal_records:
            if terminal.get("pin_anchor_evidence", False):
                continue
            point = terminal.get("point_top_left")
            if not isinstance(point, dict) or terminal.get("side") not in {"left", "right", "top", "bottom"}:
                continue
            transition_seed = dict(terminal)
            # ``point_top_left`` is the observed green skeleton pixel.  For
            # the local red-terminal transition, start at the measured body
            # boundary reference; the BFS seed below deliberately keeps the
            # skeleton pixel as its raster node.
            boundary_point = terminal.get("component_boundary_reference_top_left")
            if isinstance(boundary_point, dict):
                transition_seed["point_top_left"] = dict(boundary_point)
            transition_seeds.append(transition_seed)
        # Keep a strict seed set for raster-level edge emission.  Formal pin
        # coordinates that did not pass the local outward-skeleton test may
        # still be useful as diagnostics or for later API/OCR recovery, but
        # they must not create protocol wire pieces.
        validated_transition_seeds = [dict(anchor) for anchor in pin_anchor_by_node.values()]
        # Formal pins that did not attach to the global skeleton may still
        # have a rendered red terminal followed by a green run.  They are
        # admitted only to this local evidence gate; the helper below also
        # requires the green run to touch the existing skeleton.  The formal
        # coordinate therefore remains a hypothesis, never a wire by itself.
        known_seed_keys = {
            (
                seed.get("component_key"),
                seed.get("pin_key"),
                round(float(seed.get("point_top_left", {}).get("x", 0.0)), 1)
                if isinstance(seed.get("point_top_left"), dict)
                else None,
                round(float(seed.get("point_top_left", {}).get("y", 0.0)), 1)
                if isinstance(seed.get("point_top_left"), dict)
                else None,
            )
            for seed in transition_seeds
        }
        for formal_seed in pin_anchors:
            point = formal_seed.get("point_top_left")
            key = (
                formal_seed.get("component_key"),
                formal_seed.get("pin_key"),
                round(float(point.get("x", 0.0)), 1) if isinstance(point, dict) else None,
                round(float(point.get("y", 0.0)), 1) if isinstance(point, dict) else None,
            )
            if key not in known_seed_keys:
                transition_seeds.append(dict(formal_seed))
                known_seed_keys.add(key)
        color_transition_candidates = _local_terminal_color_segments(
            image,
            [
                seed
                for seed in transition_seeds
                if (
                    # A visible pin identity is useful metadata, but it is
                    # not a prerequisite for geometric wire recovery.  Keep
                    # OCR-unresolved component-boundary terminals in the
                    # local red/green evidence stage as well.  These records
                    # are admitted only when they came from an observed
                    # skeleton-to-boundary intersection; no library order or
                    # guessed pin number is introduced here.
                    (
                        isinstance(seed.get("pin_key"), str)
                        and bool(seed.get("pin_key"))
                    )
                    or bool(seed.get("pin_anchor_evidence"))
                    or seed.get("source") == "component_boundary_skeleton_intersection"
                    or seed.get("component_boundary_skeleton_intersection") is True
                )
            ],
            green_mask=mask,
            red_mask=red_terminal_source,
            skeleton_mask=skeleton,
            stroke_width=stroke_width,
        )
        color_transition_candidates = _filter_local_transitions_by_global_wire(
            color_transition_candidates,
            hough_base,
            stroke_width=stroke_width,
            skeleton_mask=skeleton,
            junction_records=junctions,
        )
        local_color_transition_count = len(color_transition_candidates)
        # Use the red/green transition as local endpoint evidence first.  If
        # a global Hough run already contains the local corridor, partition
        # that run at the measured transition instead of emitting a duplicate
        # copy of the same interval.
        segments = _replace_hough_terminal_overlap(
            segments,
            color_transition_candidates,
            stroke_width=stroke_width,
        )
        # The red->green observation first refines an existing global edge;
        # only the candidates that also passed the global-track gate are kept
        # as a bounded local protocol edge.  This preserves a real terminal
        # split without accepting a red-only or unconnected pin.
        partitioned_keys = {
            tuple(
                round(value, 1)
                for point in item.get("points_top_left", [])
                for value in (point.get("x"), point.get("y"))
            )
            for item in segments
            if item.get("terminal_transition_partition")
            and item.get("source") == "pin_local_color_transition"
        }
        partitioned_transition_keys = {
            tuple(
                round(value, 1)
                for point in item.get("local_transition_points_top_left", [])
                for value in (point.get("x"), point.get("y"))
            )
            for item in segments
            if item.get("local_transition_evidence")
            and isinstance(item.get("local_transition_points_top_left"), list)
        }
        for candidate in color_transition_candidates:
            key = tuple(
                round(value, 1)
                for point in candidate.get("points_top_left", [])
                for value in (point.get("x"), point.get("y"))
            )
            if key in partitioned_keys or key in partitioned_transition_keys:
                continue
            item = dict(candidate)
            item["source"] = "pin_local_color_transition"
            item["local_transition_edge_evidence"] = True
            segments.append(item)
        # Preserve the observed short raster polyline as separate protocol
        # pieces.  The target geometry can contain a real red-horizontal ->
        # one-pixel-step -> green-horizontal transition; merging it into the
        # long Hough run would erase those physical endpoints.  The helper is
        # bounded by terminal evidence and global skeleton contact, so it is
        # not a generic red-line fallback.
        raster_terminal_candidates = _local_terminal_raster_segments(
            image,
            validated_transition_seeds,
            green_mask=mask,
            red_mask=red_terminal_source,
            skeleton_mask=skeleton,
            global_segments=hough_base,
            stroke_width=stroke_width,
        )
        local_raster_terminal_candidate_count = len(raster_terminal_candidates)
        segments.extend(raster_terminal_candidates)
        color_step_candidates = _local_pin_color_step_segments(
            image,
            validated_transition_seeds,
            green_mask=mask,
            red_mask=red_terminal_source,
            skeleton_mask=skeleton,
            global_segments=hough_base,
            stroke_width=stroke_width,
        )
        local_raster_terminal_candidate_count += len(color_step_candidates)
        # A colour-step observation is redundant when another local path for
        # the same validated pin already contains the same terminal endpoint.
        # Keep it when it fills a genuine raster gap before the global line;
        # remove only endpoint-equivalent observations, never adjacent graph
        # edges with different endpoints.
        for candidate in color_step_candidates:
            candidate_bindings = candidate.get("terminal_bindings")
            candidate_binding = (
                candidate_bindings[0]
                if isinstance(candidate_bindings, list)
                and candidate_bindings
                and isinstance(candidate_bindings[0], dict)
                else None
            )
            redundant = False
            if candidate.get("local_raster_green_continuation"):
                try:
                    cfirst, csecond = _segment_points(candidate)
                except (KeyError, TypeError, ValueError):
                    continue
                orientation = candidate.get("orientation")
                if orientation == "horizontal":
                    c_axis = (cfirst[1] + csecond[1]) / 2.0
                    c_low, c_high = sorted((cfirst[0], csecond[0]))
                else:
                    c_axis = (cfirst[0] + csecond[0]) / 2.0
                    c_low, c_high = sorted((cfirst[1], csecond[1]))
                for existing in segments:
                    if existing.get("orientation") != orientation:
                        continue
                    try:
                        efirst, esecond = _segment_points(existing)
                    except (KeyError, TypeError, ValueError):
                        continue
                    e_axis = (efirst[1] + esecond[1]) / 2.0 if orientation == "horizontal" else (efirst[0] + esecond[0]) / 2.0
                    if abs(c_axis - e_axis) > max(2.0, 2.0 * stroke_width):
                        continue
                    if orientation == "horizontal":
                        e_low, e_high = sorted((efirst[0], esecond[0]))
                    else:
                        e_low, e_high = sorted((efirst[1], esecond[1]))
                    overlap = min(c_high, e_high) - max(c_low, e_low)
                    # If the already measured path starts at or before the
                    # continuation start, the short observation adds no new
                    # geometry.  If it starts later, the candidate preserves
                    # the terminal-side gap and must remain.
                    if overlap >= 0.5 and e_low <= c_low + max(0.75, float(stroke_width)):
                        extends_track_end = c_low < e_low - 0.5 or c_high > e_high + 0.5
                        if not extends_track_end:
                            redundant = True
                            break
            if candidate_binding is not None:
                try:
                    cfirst, csecond = _segment_points(candidate)
                except (KeyError, TypeError, ValueError):
                    continue
                for existing in segments:
                    bindings = existing.get("terminal_bindings")
                    first_binding = (
                        bindings[0]
                        if isinstance(bindings, list)
                        and bindings
                        and isinstance(bindings[0], dict)
                        else None
                    )
                    if not first_binding:
                        continue
                    if (
                        first_binding.get("component_key") != candidate_binding.get("component_key")
                        or first_binding.get("pin_key") != candidate_binding.get("pin_key")
                    ):
                        continue
                    if existing.get("orientation") != candidate.get("orientation"):
                        continue
                    try:
                        efirst, esecond = _segment_points(existing)
                    except (KeyError, TypeError, ValueError):
                        continue
                    candidate_axis = (cfirst[1] + csecond[1]) / 2.0 if candidate.get("orientation") == "horizontal" else (cfirst[0] + csecond[0]) / 2.0
                    existing_axis = (efirst[1] + esecond[1]) / 2.0 if candidate.get("orientation") == "horizontal" else (efirst[0] + esecond[0]) / 2.0
                    if abs(candidate_axis - existing_axis) > max(2.0, 2.0 * stroke_width):
                        continue
                    if candidate.get("orientation") == "horizontal":
                        c_low, c_high = sorted((cfirst[0], csecond[0]))
                        e_low, e_high = sorted((efirst[0], esecond[0]))
                    else:
                        c_low, c_high = sorted((cfirst[1], csecond[1]))
                        e_low, e_high = sorted((efirst[1], esecond[1]))
                    overlap = min(c_high, e_high) - max(c_low, e_low)
                    if overlap >= 0.5:
                        contained = c_low >= e_low - 0.5 and c_high <= e_high + 0.5
                        if contained or not candidate.get("local_raster_green_continuation"):
                            redundant = True
                            break
                    if (
                        not candidate.get("local_raster_green_continuation")
                        and min(abs(c_low - e_high), abs(e_low - c_high)) <= max(0.75, float(stroke_width))
                    ):
                        redundant = True
                        break
            if not redundant:
                segments.append(candidate)
        # The local BFS is allowed for every validated pin.  Its acceptance
        # gate is geometric (red-to-green transition plus skeleton/global
        # contact), so ICs are not excluded merely by component class.  Red
        # body outlines that cannot reach a green path are rejected inside
        # the bounded corridor rather than by a type-specific shortcut.
        # A formal pin coordinate alone is not a wire origin.  Restrict the
        # fused ROI BFS to anchors that already passed the local
        # pin-to-skeleton outward-path test.  Unconnected/formal-only pins
        # remain available to diagnostics and later pin identity handling,
        # but cannot create a wire candidate.
        fused_bfs_anchors = [
            anchor for anchor in pin_anchor_by_node.values()
            if isinstance(anchor, dict)
        ]
        # Add only geometric component terminals that were found by an actual
        # skeleton/boundary intersection.  A coarse bbox side is not enough
        # to start a local path.  The observed skeleton point is used as the
        # origin so this branch can recover direct-green terminal geometry
        # without fabricating a pin identity.
        fused_bfs_terminal_seeds: list[dict[str, Any]] = []
        for terminal in terminal_records:
            if terminal.get("pin_anchor_evidence", False):
                continue
            if terminal.get("source") != "component_boundary_skeleton_intersection":
                continue
            point = terminal.get("skeleton_point_top_left") or terminal.get("point_top_left")
            side = str(terminal.get("side") or "").lower()
            if not isinstance(point, dict) or side not in {"left", "right", "top", "bottom"}:
                continue
            seed = dict(terminal)
            seed["point_top_left"] = dict(point)
            seed["terminal_seed_evidence"] = "observed_component_boundary_skeleton_intersection"
            fused_bfs_terminal_seeds.append(seed)
        fused_bfs_terminal_seeds.extend(
            dict(seed) for seed in component_red_green_seeds
        )
        fused_bfs_candidates = _formal_pin_fused_transition_segments(
            image,
            fused_bfs_anchors,
            component_boxes,
            mask,
            red_terminal_source,
            skeleton_mask=skeleton,
            stroke_width=stroke_width,
            global_segments=hough_base,
            junction_records=junctions,
            terminal_seeds=fused_bfs_terminal_seeds,
        )
        fused_bfs_transition_count = len(fused_bfs_candidates)
        fused_bfs_direct_green_count = sum(
            1 for candidate in fused_bfs_candidates
            if candidate.get("local_green_terminal_evidence")
        )
        fused_bfs_terminal_seed_count = len(fused_bfs_terminal_seeds)
        # Keep the local transition as an explicit image-supported edge.  It
        # is later deduplicated only when both endpoints coincide; sharing one
        # endpoint with a Hough run is a legitimate terminal split.
        segments.extend(fused_bfs_candidates)
        # Preserve the image-supported skeleton graph only for connected
        # components that contain an observed terminal or junction.  This is
        # the bounded recall supplement: every interval still originates from
        # the green wire mask, while isolated raster fragments are not
        # promoted to protocol edges.
        observed_skeleton_segments = _evidence_connected_skeleton_candidates(
            trace_skeleton,
            skeleton_segments,
            terminal_records,
            junctions,
            stroke_width=stroke_width,
        )
        # It is retained in addition to Hough because Hough can merge away a
        # genuine bend/endpoint.  Endpoint-equivalent observations are
        # removed below; adjacent graph edges are deliberately not merged.
        for candidate in observed_skeleton_segments:
            item = dict(candidate)
            item["source"] = "wire_skeleton_local_graph_recall"
            item["skeleton_path_evidence"] = True
            item["skeleton_component_evidence"] = True
            segments.append(item)
        skeleton_recall_candidate_count = len(observed_skeleton_segments)
        # The local colour-step stage runs before BFS/skeleton candidates are
        # appended.  Repeat a binding-aware overlap check now that all local
        # observations are present.  A colour step is removed only when a
        # non-raster path for the same validated pin already covers it; a
        # short raster edge that fills a real gap remains available.
        filtered_segments: list[dict[str, Any]] = []
        for candidate in segments:
            if not (
                candidate.get("local_raster_color_step_evidence")
                or candidate.get("local_raster_green_continuation")
            ):
                filtered_segments.append(candidate)
                continue
            bindings = candidate.get("terminal_bindings")
            binding = (
                bindings[0]
                if isinstance(bindings, list)
                and bindings
                and isinstance(bindings[0], dict)
                else None
            )
            if binding is None:
                filtered_segments.append(candidate)
                continue
            try:
                cfirst, csecond = _segment_points(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            orientation = candidate.get("orientation")
            if orientation == "horizontal":
                c_axis = (cfirst[1] + csecond[1]) / 2.0
                c_low, c_high = sorted((cfirst[0], csecond[0]))
            else:
                c_axis = (cfirst[0] + csecond[0]) / 2.0
                c_low, c_high = sorted((cfirst[1], csecond[1]))
            covered = False
            for existing in segments:
                if existing is candidate or existing.get("source") == "pin_local_raster_terminal_path":
                    continue
                existing_bindings = existing.get("terminal_bindings")
                existing_binding = (
                    existing_bindings[0]
                    if isinstance(existing_bindings, list)
                    and existing_bindings
                    and isinstance(existing_bindings[0], dict)
                    else None
                )
                if not existing_binding or (
                    existing_binding.get("component_key") != binding.get("component_key")
                    or existing_binding.get("pin_key") != binding.get("pin_key")
                ):
                    continue
                if existing.get("orientation") != orientation:
                    continue
                try:
                    efirst, esecond = _segment_points(existing)
                except (KeyError, TypeError, ValueError):
                    continue
                e_axis = (efirst[1] + esecond[1]) / 2.0 if orientation == "horizontal" else (efirst[0] + esecond[0]) / 2.0
                if abs(c_axis - e_axis) > max(2.0, 2.0 * stroke_width):
                    continue
                if orientation == "horizontal":
                    e_low, e_high = sorted((efirst[0], esecond[0]))
                else:
                    e_low, e_high = sorted((efirst[1], esecond[1]))
                if min(c_high, e_high) - max(c_low, e_low) >= 0.5:
                    covered = True
                    break
            if not covered:
                filtered_segments.append(candidate)
        segments = filtered_segments
        # A component red/green seed has no pin identity and can sometimes
        # rediscover a short interval already represented by a global Hough
        # or skeleton path.  Remove only a fully contained, same-axis copy;
        # an unbound local interval that extends beyond the existing track is
        # retained because it may be the only measured terminal segment.
        local_unique: list[dict[str, Any]] = []
        for candidate in segments:
            if candidate.get("source") != "pin_local_fused_bfs_transition":
                local_unique.append(candidate)
                continue
            bindings = candidate.get("terminal_bindings")
            has_identity = any(
                isinstance(binding, dict) and binding.get("component_key") is not None
                for binding in (bindings if isinstance(bindings, list) else [])
            )
            if has_identity:
                local_unique.append(candidate)
                continue
            try:
                cfirst, csecond = _segment_points(candidate)
            except (KeyError, TypeError, ValueError):
                continue
            orientation = candidate.get("orientation")
            if orientation == "horizontal":
                c_axis = (cfirst[1] + csecond[1]) / 2.0
                c_low, c_high = sorted((cfirst[0], csecond[0]))
            elif orientation == "vertical":
                c_axis = (cfirst[0] + csecond[0]) / 2.0
                c_low, c_high = sorted((cfirst[1], csecond[1]))
            else:
                local_unique.append(candidate)
                continue
            covered = False
            for existing in segments:
                if existing is candidate or existing.get("source") == "pin_local_fused_bfs_transition":
                    continue
                if existing.get("orientation") != orientation:
                    continue
                try:
                    efirst, esecond = _segment_points(existing)
                except (KeyError, TypeError, ValueError):
                    continue
                if orientation == "horizontal":
                    e_axis = (efirst[1] + esecond[1]) / 2.0
                    e_low, e_high = sorted((efirst[0], esecond[0]))
                else:
                    e_axis = (efirst[0] + esecond[0]) / 2.0
                    e_low, e_high = sorted((efirst[1], esecond[1]))
                if (
                    abs(c_axis - e_axis) <= max(2.0, 2.0 * stroke_width)
                    and e_low - 1.0 <= c_low
                    and c_high <= e_high + 1.0
                ):
                    covered = True
                    break
            if not covered:
                local_unique.append(candidate)
        segments = local_unique
        dedup_tolerance = max(0.75, min(1.5, 0.5 * max(1.0, float(stroke_width))))
        endpoint_dedup_tolerance = round(float(dedup_tolerance), 3)
        segments = _deduplicate_endpoint_equivalent_segments(
            segments,
            tolerance=dedup_tolerance,
        )
        # Local pin/ROI fusion can union a short terminal-origin path with a
        # longer Hough interval after the first observed-node split.  Apply
        # the split once more after *all* image-supported candidates have
        # been merged so a real terminal or junction remains a graph node in
        # the final edge list.  This partitions an existing segment only; it
        # does not create a path from a bbox, library order, or edge count.
        terminal_split_candidates = [
            segment
            for segment in segments
            if "hough" in str(segment.get("source") or "")
        ]
        non_terminal_split_candidates = [
            segment
            for segment in segments
            if "hough" not in str(segment.get("source") or "")
        ]
        # Some formal pins pass the local red/green validation but do not
        # become entries in ``terminal_records`` because their one-pixel
        # green contact was recovered only after the initial terminal scan.
        # They are still legitimate geometric split nodes for an already
        # supported Hough path.  Promote only bindings witnessed by the
        # local transition/BFS stage; an unvalidated formal pin never creates
        # a wire or a split by itself.
        validated_split_keys: set[tuple[Any, Any]] = set()
        for local_candidate in [
            *color_transition_candidates,
            *color_step_candidates,
            *fused_bfs_candidates,
        ]:
            bindings = local_candidate.get("terminal_bindings")
            if not isinstance(bindings, list) or not bindings:
                continue
            binding = bindings[0]
            if not isinstance(binding, dict):
                continue
            key = (binding.get("component_key"), binding.get("pin_key"))
            if key[0] is not None and key[1] is not None:
                validated_split_keys.add(key)
        split_terminal_records = list(
            record
            for record in terminal_records
            if record.get("pin_anchor_evidence", False)
        )
        known_split_keys = {
            (record.get("component_key"), record.get("pin_key"))
            for record in split_terminal_records
        }
        for anchor in pin_anchors:
            if not isinstance(anchor, dict):
                continue
            key = (anchor.get("component_key"), anchor.get("pin_key"))
            point = anchor.get("point_top_left")
            if key not in validated_split_keys or key in known_split_keys:
                continue
            if not isinstance(point, dict):
                continue
            record = dict(anchor)
            record["pin_anchor_evidence"] = True
            record["source"] = "validated_formal_pin_local_transition"
            split_terminal_records.append(record)
            known_split_keys.add(key)
        terminal_split_candidates = _split_at_observed_nodes(
            terminal_split_candidates,
            split_terminal_records,
            [
                record
                for record in junctions
                if record.get("classification") == "junction_candidate"
            ],
            stroke_width=stroke_width,
        )
        segments = terminal_split_candidates + non_terminal_split_candidates
        # Local pin/ROI fusion above can rejoin adjacent Hough children.  Run
        # the measured colour-transition split once after that fusion so a
        # real red/green transition remains a graph boundary in the final
        # candidate set.  This is still only a split of an existing Hough
        # path; it does not promote red pixels to global wire geometry.
        hough_transition_candidates = [
            segment
            for segment in segments
            if "hough" in str(segment.get("source") or "")
        ]
        non_hough_transition_candidates = [
            segment
            for segment in segments
            if "hough" not in str(segment.get("source") or "")
        ]
        hough_transition_candidates, late_color_points = _split_hough_at_local_color_transitions(
            image,
            hough_transition_candidates,
            terminal_records,
            green_mask=mask,
            red_mask=red_terminal_source,
            stroke_width=stroke_width,
        )
        color_transition_split_points.extend(late_color_points)
        segments = hough_transition_candidates + non_hough_transition_candidates
        for segment in segments:
            if segment.get("source") not in {
                "wire_skeleton_short_evidence",
                "pin_local_color_transition",
                "wire_skeleton_local_graph_recall",
                "wire_hough_with_pin_local_path",
                "pin_roi_fused_graph_recall",
                "non_ic_red_body_local_path",
                "pin_local_connected_red_terminal",
                "pin_local_red_green_transition",
                "wire_hough_local_red_gap",
                "pin_local_raster_terminal_path",
                "pin_local_fused_bfs_transition",
                "wire_red_gap_track_union",
                "wire_red_terminal_parent_track",
                "wire_observed_node_parent_track",
            }:
                segment["source"] = "wire_skeleton_validated_hough"
    else:
        segments = skeleton_segments
    # In the coloured renderer, a formal pin point is a local anchor, not a
    # licence to move an independently detected long Hough endpoint.  Pin
    # snapping is therefore kept for the local transition records only; the
    # global candidate remains at its own image-supported endpoint.  Generic
    # component/skeleton terminals can still snap global endpoints.
    terminal_snap_records = terminal_records
    if colored_wire_layer:
        terminal_snap_records = []
        # Green-mode global Hough endpoints remain image-native.  Coarse
        # component-boundary records are retained for local evidence and
        # topology, but are not allowed to move an unrelated Hough endpoint.
    segments = _snap_segment_terminals(segments, terminal_snap_records)
    segments = _snap_segment_junctions(segments, junctions, stroke_width=stroke_width)
    if colored_wire_layer:
        # The global Hough path remains authoritative for the wire interval;
        # only its already-bound endpoint may be projected to the matching
        # measured pin terminal.  Local red/green candidates stay untouched.
        segments = _snap_bound_hough_terminals(
            segments,
            terminal_records,
            stroke_width=stroke_width,
        )
    segments = _remove_isolated_short_segments(segments)
    if not colored_wire_layer:
        segments = _merge_collinear_paths(segments, junctions)
    segments = _remove_degenerate_segments(segments)
    segments = _post_geometry_ocr_evidence(segments, ocr_boxes)
    segments = _remove_black_text_like_segments(
        segments,
        colored_wire_layer=colored_wire_layer,
    )
    if colored_wire_layer:
        protected_points: list[tuple[float, float]] = []
        for record in terminal_records:
            if not (
                record.get("pin_anchor_evidence", False)
                or record.get("source") == "component_boundary_skeleton_intersection"
            ):
                continue
            point = record.get("point_top_left") or record.get("component_boundary_reference_top_left")
            if isinstance(point, dict):
                try:
                    protected_points.append((float(point["x"]), float(point["y"])))
                except (KeyError, TypeError, ValueError):
                    pass
        for record in junctions:
            point = record.get("point_top_left")
            if isinstance(point, dict):
                try:
                    protected_points.append((float(point["x"]), float(point["y"])))
                except (KeyError, TypeError, ValueError):
                    pass
        # A validated red/green local transition is a measured graph node,
        # even when it is not classified as an electrical junction.  The
        # final collinear compactor must not join the Hough child on one side
        # back across this point, otherwise the terminal split recovered by
        # the local ROI is silently erased.
        for segment in segments:
            if not segment.get("terminal_transition_partition"):
                continue
            for point in segment.get("points_top_left", []):
                if not isinstance(point, dict):
                    continue
                try:
                    protected_points.append((float(point["x"]), float(point["y"])))
                except (KeyError, TypeError, ValueError):
                    pass
        protected_points.extend(color_transition_split_points)
        segments = _compact_global_wire_fragments(
            segments,
            protected_points=protected_points,
            stroke_width=stroke_width,
            support_mask=mask,
        )
        # Apply centre-line correction after all topology compaction.  The
        # operation is perpendicular-only and therefore cannot undo a
        # terminal/junction split that the preceding stages established.
        segments, axis_centerline_refinement_count = _refine_axis_coordinate_from_mask(
            segments,
            mask,
            stroke_width=stroke_width,
        )
    # The source geometry is a raster.  Quantize the final observed points to
    # the nearest pixel so a sub-pixel skeleton approximation cannot turn a
    # genuinely 5-pixel endpoint match into a 5.1-pixel miss.
    for segment in segments:
        points = segment.get("points_top_left")
        if not isinstance(points, list) or len(points) != 2:
            continue
        for point in points:
            if isinstance(point, dict):
                for key in ("x", "y"):
                    if key in point:
                        point[key] = int(round(float(point[key])))
    for index, segment in enumerate(segments, start=1):
        segment["id"] = f"wire_{index}"
    return {
        "segments_top_left": segments,
        "component_terminals_top_left": terminal_records,
        "junction_candidates_top_left": junctions,
        "skeleton_pixel_count": int(cv2.countNonZero(skeleton)),
        "wire_mask_pixel_count": int(cv2.countNonZero(mask)),
        "component_mask_count": len(component_boxes),
        "pin_anchor_count": len(pin_anchors),
        "pin_anchor_connected_count": len(pin_anchor_by_node),
        "pin_anchor_recovery_count": pin_anchor_recovery_count,
        "skeleton_gap_repair_count": skeleton_gap_repair_count,
        "hough_candidate_count": len(segments) if colored_wire_layer else 0,
        "local_color_transition_candidate_count": local_color_transition_count,
        "local_raster_terminal_candidate_count": local_raster_terminal_candidate_count,
        "fused_bfs_transition_count": fused_bfs_transition_count,
        "fused_bfs_direct_green_count": fused_bfs_direct_green_count,
        "fused_bfs_terminal_seed_count": fused_bfs_terminal_seed_count,
        "component_red_green_seed_count": component_red_green_seed_count,
        "red_gap_track_union_count": red_gap_track_union_count,
        "red_terminal_parent_track_count": red_terminal_parent_track_count,
        "observed_node_parent_track_count": observed_node_parent_track_count,
        "red_terminal_overlap_union_count": red_terminal_overlap_union_count,
        "pin_local_skeleton_candidate_count": pin_local_skeleton_candidate_count,
        "fused_roi_candidate_count": fused_roi_candidate_count,
        "formal_pin_roi_candidate_count": formal_pin_roi_candidate_count,
        "formal_pin_roi_recovered_count": formal_pin_roi_recovered_count,
        "local_terminal_snap_count": local_terminal_snap_count,
        "connected_red_candidate_count": connected_red_candidate_count,
        "validated_red_green_candidate_count": validated_red_green_candidate_count,
        "local_red_gap_merge_count": local_red_gap_merge_count,
        "skeleton_recall_candidate_count": skeleton_recall_candidate_count,
        "endpoint_dedup_tolerance_px": endpoint_dedup_tolerance,
        "axis_centerline_refinement_count": axis_centerline_refinement_count,
        "wire_path_backend": "mask_hough_with_skeleton_evidence" if colored_wire_layer else "skeleton_path",
        "ocr_box_count": len(ocr_boxes),
        "colored_wire_layer": colored_wire_layer,
        "wire_layer_mode": "green" if colored_wire_layer else "black_achromatic",
        "pin_local_source": "green" if colored_wire_layer else "black_achromatic",
        "red_local_terminal_source": bool(colored_wire_layer),
        "estimated_stroke_width_px": round(stroke_width, 3),
        "morphology_close_size": close_size,
        "high_resolution_mode": high_resolution,
        "skeleton_neighbor_model": "4-connected",
        "note": (
            "OCR is posterior evidence only; terminal coordinates are projected "
            "to the component boundary and remain independent of pin identity."
        ),
    }


def segments_to_submission_edges(
    segments: Iterable[dict[str, Any]],
    image_height: float,
) -> dict[str, list[dict[str, float]]]:
    """Convert two-point top-left segments to contest ``edges`` format.

    The official protocol requires every edge to contain exactly two points.
    Bends are already split by ``_path_to_segments`` and therefore remain
    separate ``edge_i`` entries here.
    """
    from common.coordinate import top_left_point_to_bottom_left

    edges: dict[str, list[dict[str, float]]] = {}
    for index, segment in enumerate(segments, start=1):
        points = segment.get("points_top_left")
        if not isinstance(points, list) or len(points) != 2:
            continue
        p1, p2 = _point(points[0]), _point(points[1])
        edges[f"edge_{index}"] = [
            top_left_point_to_bottom_left(p1, image_height, decimals=2),
            top_left_point_to_bottom_left(p2, image_height, decimals=2),
        ]
    return edges
