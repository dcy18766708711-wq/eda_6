"""Find pin tips from observed wire-terminal image geometry.

No fixed terminal offset is used here.  A candidate is accepted only when a
wire segment reaches the corresponding IC side and local dark pixels support
the same terminal axis.  The returned point remains in top-left coordinates.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

try:  # Support package and direct-module execution.
    from .line_detector import generate_pin_candidates  # type: ignore
except ImportError:
    from topology.line_detector import generate_pin_candidates  # type: ignore


VALID_SOURCES = {
    "local_terminal_wire_transition",
    "local_terminal_intersection",
    "skeleton_terminal_intersection",
    "wire_terminal_intersection",
    "terminal_boundary",
    "pin_pin_overlap",
}


def _points(segment: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    values = segment["points_top_left"]
    return (
        (float(values[0]["x"]), float(values[0]["y"])),
        (float(values[1]["x"]), float(values[1]["y"])),
    )


def _dark(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return gray < 210


def _thin(mask: np.ndarray) -> np.ndarray:
    """Thin a small local ROI without requiring global wire reconstruction."""
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        return cv2.ximgproc.thinning(binary, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    skeleton = np.zeros_like(binary)
    work = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(work):
        opened = cv2.morphologyEx(work, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(work, opened))
        work = cv2.erode(work, element)
    return skeleton


def _adaptive_roi_margin(bbox: list[float], image_shape: tuple[int, ...]) -> int:
    """Choose a local exterior band from the visible IC scale.

    A fixed 40--80 px band is not stable across the public images.  The
    smaller IC dimension is a useful local scale; the clamp prevents nearby
    labels from dominating small symbols and retains long terminal stubs on
    high-resolution drawings.
    """
    x1, y1, x2, y2 = map(float, bbox)
    scale = min(max(1.0, x2 - x1), max(1.0, y2 - y1))
    return int(round(max(20.0, min(96.0, 0.08 * scale))))


def _estimate_line_width(mask: np.ndarray, skeleton: np.ndarray) -> float:
    """Estimate local raster line width without case-specific constants."""
    if not np.any(mask) or not np.any(skeleton):
        return 1.0
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    radii = distance[skeleton > 0]
    if radii.size == 0:
        return 1.0
    # Large body interiors are removed before skeletonization; clipping still
    # prevents an unusually thick title glyph from widening terminal clusters.
    return float(np.clip(2.0 * np.median(radii), 1.0, 4.0))


def _cluster_terminal_rows(
    raw: list[tuple[float, float, float]],
    *,
    side: str,
    bbox: list[float],
    line_width: float,
) -> list[dict[str, Any]]:
    """Merge raster-thickness duplicates while retaining separate pin rows."""
    if not raw:
        return []
    x1, y1, x2, y2 = map(float, bbox)
    raw.sort(key=lambda item: item[0])
    groups: list[list[tuple[float, float, float]]] = []
    merge_distance = max(2.5, 2.5 * float(line_width))
    for item in raw:
        if not groups or item[0] - groups[-1][-1][0] > merge_distance:
            groups.append([item])
        else:
            groups[-1].append(item)
    candidates: list[dict[str, Any]] = []
    for group in groups:
        # A true pin stroke is only a few raster rows/columns thick.  A wide
        # continuous group is normally an IC outline or a nearby parallel
        # wire, not dozens of terminals merged into one candidate.
        if group[-1][0] - group[0][0] > max(8.0, 4.0 * float(line_width)):
            continue
        weight = max(1e-6, sum(item[2] for item in group))
        coordinate = sum(item[0] * item[2] for item in group) / weight
        tip_axis = sum(item[1] * item[2] for item in group) / weight
        # Corner strokes are usually the IC outline, not a terminal.  Scale
        # the exclusion margin for narrow R/C/V symbols; a fixed 6 px margin
        # would remove every scan position from an 8 px-wide resistor box.
        edge_margin_x = max(1.0, min(6.0, 0.25 * (x2 - x1)))
        edge_margin_y = max(1.0, min(6.0, 0.25 * (y2 - y1)))
        if side in {"left", "right"} and not (y1 + edge_margin_y < coordinate < y2 - edge_margin_y):
            continue
        if side in {"top", "bottom"} and not (x1 + edge_margin_x < coordinate < x2 - edge_margin_x):
            continue
        point = {"x": tip_axis, "y": coordinate} if side in {"left", "right"} else {"x": coordinate, "y": tip_axis}
        boundary = x1 if side == "left" else x2 if side == "right" else y1 if side == "top" else y2
        measured_axis = point["x"] if side in {"left", "right"} else point["y"]
        candidates.append(
            {
                "side": side,
                "point_top_left": {"x": round(point["x"], 2), "y": round(point["y"], 2)},
                "evidence_score": round(min(1.0, sum(item[2] for item in group) / len(group)), 4),
                "source": "local_roi_bfs_terminal_path",
                "tip_source": "local_terminal_wire_transition",
                "tip_evidence": 1.0,
                "boundary_distance_px": round(abs(measured_axis - boundary), 3),
                "local_evidence": {
                    "outward_skeleton_path": True,
                    "terminal_colour_transition": True,
                    "support_rows": len(group),
                    "estimated_line_width_px": round(float(line_width), 3),
                },
            }
        )
    return candidates


def detect_local_terminal_candidates(
    image: np.ndarray,
    bbox_top_left: list[float],
    *,
    ocr_items: list[dict[str, Any]] | None = None,
    minimum_reach: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure IC pin tips from local terminal--wire intersections.

    This intentionally does not use a whole-page wire graph or a calibrated
    bbox offset.  For each side of the refined IC body, a local skeleton must
    visibly connect the body-boundary seed to the exterior band.  When the
    raster has separate terminal/net colours, the outer endpoint of the
    observed terminal stroke is used as the terminal--wire transition; a
    monochrome image falls back to the measured outer skeleton point.
    """
    x1, y1, x2, y2 = map(float, bbox_top_left)
    height, width = image.shape[:2]
    margin = _adaptive_roi_margin(bbox_top_left, image.shape)
    left, top = max(0, int(math.floor(x1 - margin))), max(0, int(math.floor(y1 - margin)))
    right, bottom = min(width, int(math.ceil(x2 + margin))), min(height, int(math.ceil(y2 + margin)))
    if right <= left or bottom <= top:
        return [], {"roi_top_left": [left, top, right, bottom], "margin_px": margin, "reason": "empty_roi"}

    crop = image[top:bottom, left:right]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # In the public editor raster, terminal strokes are red and nets are
    # green.  Using only these layers prevents black OCR glyphs from creating
    # fake BFS paths. Monochrome exports retain the dark-ink fallback.
    if crop.ndim == 3:
        blue, green, red = (channel.astype(np.int16) for channel in cv2.split(crop))
        terminal_red = (red >= green + 25) & (red >= blue + 25) & (red >= 100)
        wire_green = (green >= red + 25) & (green >= blue + 25) & (green >= 100)
        colour_mask = terminal_red | wire_green
        mask = np.where(colour_mask if int(colour_mask.sum()) >= 20 else gray < 210, 255, 0).astype(np.uint8)
        # Red terminal strokes and green wires are rasterised independently.
        # A one-pixel low-saturation anti-alias gap must not break an otherwise
        # measured local path.  Close only one-pixel axial gaps; do not use a
        # wide dilation that could manufacture a terminal or join wires.
        mask = cv2.bitwise_or(
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)),
            cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 3), np.uint8)),
        )
    else:
        terminal_red = np.zeros_like(gray, dtype=bool)
        mask = np.where(gray < 210, 255, 0).astype(np.uint8)
    ix1, iy1, ix2, iy2 = int(round(x1 - left)), int(round(y1 - top)), int(round(x2 - left)), int(round(y2 - top))
    cv2.rectangle(mask, (max(0, ix1 + 2), max(0, iy1 + 2)), (min(mask.shape[1] - 1, ix2 - 2), min(mask.shape[0] - 1, iy2 - 2)), 0, thickness=-1)
    # OCR is deliberately not erased here. Pin number/name boxes frequently
    # overlap a 6--10 px terminal stub; deleting them before geometry causes
    # systematic pin recall loss. OCR is bound only after a terminal path is
    # measured below.
    skeleton = _thin(mask)
    line_width = _estimate_line_width(mask, skeleton)


    raw_by_side: dict[str, list[tuple[float, float, float]]] = {side: [] for side in ("left", "right", "top", "bottom")}
    inset_x = max(1, min(6, int(round(0.015 * (x2 - x1)))))
    inset_y = max(1, min(6, int(round(0.015 * (y2 - y1)))))
    reach = max(int(minimum_reach), min(12, margin // 3))
    # Let a path follow a real 1--4 px wire stroke without merging pin rows;
    # its width is measured from the local raster rather than fixed for one
    # editor style or resolution.
    tangent = max(1, min(4, int(math.ceil(line_width))))

    def outward_path(
        side: str,
        coordinate: int,
        *,
        tangent_width: int | None = None,
    ) -> list[tuple[int, int]] | None:
        """BFS in a narrow normal band; co-occurring pixels are insufficient."""
        local_tangent = tangent if tangent_width is None else max(1, int(tangent_width))
        if side in {"left", "right"}:
            cy = coordinate - top
            ylo, yhi = max(0, cy - local_tangent), min(skeleton.shape[0], cy + local_tangent + 1)
            if side == "left":
                xlo, xhi = max(0, ix1 - margin), min(skeleton.shape[1], ix1 + 3)
                reached = lambda x, _y: x <= ix1 - reach
                seeds = [(x, y) for y in range(ylo, yhi) for x in range(max(xlo, ix1 - 2), xhi) if skeleton[y, x]]
            else:
                xlo, xhi = max(0, ix2 - 2), min(skeleton.shape[1], ix2 + margin + 1)
                reached = lambda x, _y: x >= ix2 + reach
                seeds = [(x, y) for y in range(ylo, yhi) for x in range(xlo, min(xhi, ix2 + 3)) if skeleton[y, x]]
        else:
            cx = coordinate - left
            xlo, xhi = max(0, cx - local_tangent), min(skeleton.shape[1], cx + local_tangent + 1)
            if side == "top":
                ylo, yhi = max(0, iy1 - margin), min(skeleton.shape[0], iy1 + 3)
                reached = lambda _x, y: y <= iy1 - reach
                seeds = [(x, y) for y in range(max(ylo, iy1 - 2), yhi) for x in range(xlo, xhi) if skeleton[y, x]]
            else:
                ylo, yhi = max(0, iy2 - 2), min(skeleton.shape[0], iy2 + margin + 1)
                reached = lambda _x, y: y >= iy2 + reach
                seeds = [(x, y) for y in range(ylo, min(yhi, iy2 + 3)) for x in range(xlo, xhi) if skeleton[y, x]]
        queue = list(dict.fromkeys(seeds))
        seen = set(queue)
        for x, y in queue:
            if reached(x, y):
                return queue
        index = 0
        while index < len(queue):
            x, y = queue[index]
            index += 1
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if nx < xlo or nx >= xhi or ny < ylo or ny >= yhi or (nx, ny) in seen or not skeleton[ny, nx]:
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny))
                if reached(nx, ny):
                    return queue
        return None

    def terminal_transition(side: str, coordinate: int, path: list[tuple[int, int]]) -> float | None:
        """Use the observed red pin-stroke / net-layer transition as the tip."""
        if side in {"left", "right"}:
            cy = coordinate - top
            values = terminal_red[max(0, cy - 2):min(terminal_red.shape[0], cy + 3)].any(axis=0)
            indices = np.flatnonzero(values)
            boundary = ix1 if side == "left" else ix2
        else:
            cx = coordinate - left
            values = terminal_red[:, max(0, cx - 2):min(terminal_red.shape[1], cx + 3)].any(axis=1)
            indices = np.flatnonzero(values)
            boundary = iy1 if side == "top" else iy2
        runs: list[np.ndarray] = []
        for value in indices:
            if not runs or value > runs[-1][-1] + 1:
                runs.append(np.asarray([value]))
            else:
                runs[-1] = np.append(runs[-1], value)
        touching = [run for run in runs if run.min() <= boundary + 2 and run.max() >= boundary - 2]
        if touching:
            run = max(touching, key=len)
            tip = int(run.min()) if side in {"left", "top"} else int(run.max())
            return float((left + tip) if side in {"left", "right"} else (top + tip))
        # Monochrome fallback: retain a measured outer path pixel, but mark
        # it lower confidence through the caller's score.
        axis_values = [point[0] if side in {"left", "right"} else point[1] for point in path]
        if not axis_values:
            return None
        return float(left + min(axis_values) if side == "left" else left + max(axis_values) if side == "right" else top + min(axis_values) if side == "top" else top + max(axis_values))

    # Scan each side independently.  Unlike the old raster-band test, every
    # accepted hit has an actual BFS path from the IC boundary to the exterior.
    for y in range(max(top, int(y1) + inset_y), min(bottom, int(y2) - inset_y)):
        for side in ("left", "right"):
            path = outward_path(side, y)
            if path is None:
                continue
            tip = terminal_transition(side, y, path)
            if tip is not None:
                raw_by_side[side].append((float(y), tip, 1.0))
    for x in range(max(left, int(x1) + inset_x), min(right, int(x2) - inset_x)):
        for side in ("top", "bottom"):
            path = outward_path(side, x)
            if path is None:
                continue
            tip = terminal_transition(side, x, path)
            if tip is not None:
                raw_by_side[side].append((float(x), tip, 1.0))

    # The legacy scanner is used only to propose observed side axes when a
    # one-pixel skeleton seed was missed.  Its old bbox-offset point is never
    # kept: every accepted rescue axis must still pass the local BFS and the
    # terminal transition measurement above.
    scan_anchors = generate_pin_candidates(
        image,
        bbox_top_left,
        margin=max(16, margin),
        expected_offset=max(4.0, 2.0 * line_width),
    )
    rescue_hits = 0
    for anchor in scan_anchors:
        side = str(anchor.get("side", ""))
        point = anchor.get("point_top_left", {})
        if side not in raw_by_side or not isinstance(point, dict):
            continue
        axis = point.get("y") if side in {"left", "right"} else point.get("x")
        try:
            coordinate = int(round(float(axis)))
        except (TypeError, ValueError):
            continue
        # First try the normal measured band, then a slightly wider local
        # band for anti-aliased/segmented strokes. It remains much narrower
        # than pin pitch and cannot use a remote parallel wire as evidence.
        path = outward_path(side, coordinate)
        if path is None:
            path = outward_path(side, coordinate, tangent_width=min(5, tangent + 1))
        if path is None:
            continue
        tip = terminal_transition(side, coordinate, path)
        if tip is None:
            continue
        raw_by_side[side].append((float(coordinate), tip, 0.85))
        rescue_hits += 1

    candidates = [
        candidate
        for side in raw_by_side
        for candidate in _cluster_terminal_rows(
            raw_by_side[side],
            side=side,
            bbox=bbox_top_left,
            line_width=line_width,
        )
    ]
    candidates.sort(key=lambda item: (item["side"], item["point_top_left"]["y"], item["point_top_left"]["x"]))
    return candidates, {
        "roi_top_left": [left, top, right, bottom],
        "margin_px": margin,
        "estimated_line_width_px": round(line_width, 3),
        "raw_terminal_hits": {side: len(values) for side, values in raw_by_side.items()},
        "scan_anchor_count": len(scan_anchors),
        "scan_anchor_rescue_hits": rescue_hits,
        "connectivity": 4,
    }


def _wire_endpoint(
    candidate: dict[str, Any],
    bbox: list[float],
    segments: list[dict[str, Any]],
) -> tuple[tuple[float, float], str] | None:
    side = candidate.get("side")
    axis = candidate["point_top_left"]
    x1, y1, x2, y2 = map(float, bbox)
    matches: list[tuple[float, tuple[float, float], str]] = []
    for segment in segments:
        p1, p2 = _points(segment)
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        horizontal = abs(dy) <= 3.0 and abs(dx) >= 4.0
        vertical = abs(dx) <= 3.0 and abs(dy) >= 4.0
        if side in {"left", "right"} and not horizontal:
            continue
        if side in {"top", "bottom"} and not vertical:
            continue
        if side in {"left", "right"}:
            y = (p1[1] + p2[1]) / 2.0
            if abs(y - float(axis["y"])) > 7.0:
                continue
            outside = max(p1[0], p2[0]) < x1 + 8.0 if side == "left" else min(p1[0], p2[0]) > x2 - 8.0
            if not outside:
                continue
            point = (max(p1[0], p2[0]), y) if side == "left" else (min(p1[0], p2[0]), y)
            distance = abs(point[0] - float(axis["x"])) + abs(y - float(axis["y"]))
        else:
            x = (p1[0] + p2[0]) / 2.0
            if abs(x - float(axis["x"])) > 7.0:
                continue
            outside = max(p1[1], p2[1]) < y1 + 8.0 if side == "top" else min(p1[1], p2[1]) > y2 - 8.0
            if not outside:
                continue
            point = (x, max(p1[1], p2[1])) if side == "top" else (x, min(p1[1], p2[1]))
            distance = abs(x - float(axis["x"])) + abs(point[1] - float(axis["y"]))
        matches.append((distance, point, str(segment.get("source", ""))))
    if not matches:
        return None
    _distance, point, source = min(matches, key=lambda item: item[0])
    return point, source


def _pixel_boundary(
    dark: np.ndarray,
    candidate: dict[str, Any],
    bbox: list[float],
    endpoint: tuple[float, float],
) -> tuple[float, float] | None:
    side = candidate.get("side")
    x1, y1, x2, y2 = map(float, bbox)
    height, width = dark.shape[:2]
    x, y = endpoint
    if side in {"left", "right"}:
        y0 = max(0, int(round(y)) - 3)
        y1i = min(height, int(round(y)) + 4)
        lo = max(0, int(round(x1 - 48))) if side == "left" else max(0, int(round(x2 - 8)))
        hi = min(width, int(round(x1 + 8))) if side == "left" else min(width, int(round(x2 + 48)))
        points = np.argwhere(dark[y0:y1i, lo:hi])
        if points.size == 0:
            return endpoint
        # Use the outermost observed terminal/wire pixel in the direction of
        # the component.  This is a measured image point, not a calibrated
        # offset from the bbox.
        if side == "left":
            px = float(lo + points[:, 1].max())
        else:
            px = float(lo + points[:, 1].min())
        return px, float(y)
    x0 = max(0, int(round(x)) - 3)
    x1i = min(width, int(round(x)) + 4)
    lo = max(0, int(round(y1 - 48))) if side == "top" else max(0, int(round(y2 - 8)))
    hi = min(height, int(round(y1 + 8))) if side == "top" else min(height, int(round(y2 + 48)))
    points = np.argwhere(dark[lo:hi, x0:x1i])
    if points.size == 0:
        return endpoint
    if side == "top":
        py = float(lo + points[:, 0].max())
    else:
        py = float(lo + points[:, 0].min())
    return float(x), py


def _refine_endpoint_from_pixels(
    dark: np.ndarray,
    candidate: dict[str, Any],
    bbox: list[float],
    endpoint: tuple[float, float],
) -> tuple[float, float]:
    """Move a wire endpoint to the measured terminal-to-wire transition."""
    side = candidate.get("side")
    x, y = endpoint
    height, width = dark.shape[:2]
    if side in {"left", "right"}:
        cy = int(round(y))
        y0, y1i = max(0, cy - 2), min(height, cy + 3)
        if side == "left":
            # Start at the measured wire endpoint.  Starting at the bbox
            # edge can skip the terminal ring entirely when the ring is
            # separated from the body outline by anti-aliasing.
            start = min(width - 1, int(round(x)))
            stop = max(-1, start - 28)
            columns = list(range(start, stop, -1))
        else:
            start = max(0, int(round(x)))
            stop = min(width, start + 28)
            columns = list(range(start, stop))
        counts = [int(dark[y0:y1i, column].sum()) for column in columns]
        for index in range(1, len(columns) - 5):
            previous, current = counts[index - 1], counts[index]
            persistent = all(value >= 1 for value in counts[index : index + 5])
            if previous >= 4 and 2 <= current <= 3 and persistent:
                shift = max(2, min(3, int(round((previous - current) / 2.0)) + 1))
                return float(columns[index + shift]), y
        return x, y
    # For top/bottom terminals, the open terminal shape is wider than the
    # wire.  The first stable narrow run after that width transition is a
    # measured image feature and is more faithful than bbox + offset.
    x1, y1, x2, y2 = map(float, bbox)
    cx = int(round(x))
    x0, x1i = max(0, cx - 2), min(width, cx + 3)
    if side == "bottom":
        start = max(0, int(math.floor(y2)))
        stop = min(height, start + 28)
        rows = list(range(start, stop))
    else:
        start = min(height - 1, int(math.ceil(y1)))
        stop = max(-1, start - 28)
        rows = list(range(start, stop, -1))
    counts = [int(dark[row, x0:x1i].sum()) for row in rows]
    for index in range(1, len(rows) - 5):
        previous, current = counts[index - 1], counts[index]
        persistent = all(value >= 2 for value in counts[index : index + 5])
        if previous >= 4 and 2 <= current <= 3 and persistent:
            # Move into the stable wire stroke by half of the observed
            # width drop, capped by the local 5-pixel raster window.
            shift = max(1, min(3, int(round((previous - current) / 2.0))))
            return x, float(rows[index + shift])
    return x, y


def refine_pin_candidates_with_wires(
    image: np.ndarray,
    bbox_top_left: list[float],
    candidates: list[dict[str, Any]],
    wire_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach measured wire-terminal points to side-local pin candidates."""
    dark = _dark(image)
    refined: list[dict[str, Any]] = []
    for original in candidates:
        candidate = dict(original)
        endpoint_match = _wire_endpoint(candidate, bbox_top_left, wire_segments)
        if endpoint_match is None:
            candidate["tip_source"] = "unresolved_no_wire_evidence"
            refined.append(candidate)
            continue
        endpoint, wire_source = endpoint_match
        # The morphology/Hough segment has been clipped at the observed wire
        # endpoint nearest the component.  Use that measured endpoint.  A
        # scan of the whole bbox side would select the opposite edge of a
        # hollow terminal circle and would not be the wire/pin intersection.
        # A skeleton path is already one pixel wide and has been split at the
        # observed component-terminal boundary.  Its endpoint is therefore
        # the actual submitted pin tip.  Do not nudge it with a fixed raster
        # offset: that would make the 5 px pin metric resolution-dependent.
        if wire_source == "wire_skeleton_path":
            point = endpoint
            tip_source = "skeleton_terminal_intersection"
        else:
            # Compatibility only for callers still supplying legacy Hough
            # segments.  The primary wire detector no longer emits these.
            point = _refine_endpoint_from_pixels(dark, candidate, bbox_top_left, endpoint)
            tip_source = "wire_terminal_intersection"
        candidate["point_top_left"] = {"x": round(point[0], 2), "y": round(point[1], 2)}
        candidate["wire_endpoint_top_left"] = {
            "x": round(endpoint[0], 2),
            "y": round(endpoint[1], 2),
        }
        candidate["tip_source"] = tip_source
        candidate["tip_evidence"] = 1.0
        refined.append(candidate)
    return refined
