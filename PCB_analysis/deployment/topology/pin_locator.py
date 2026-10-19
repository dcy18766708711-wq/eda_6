#!/usr/bin/env python3
"""Locate IC pin points from image geometry and bind explicit OCR evidence.

Image geometry determines each pin-tip coordinate.  The IC library is only a
semantic consistency aid for OCR text; it must neither create, discard nor
relocate geometric pin candidates.  OCR can assign a visible pin number/name,
but is never trusted to provide the final coordinate.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.coordinate import (  # noqa: E402
    bottom_left_bbox_to_top_left,
    top_left_point_to_bottom_left,
)
from ic_library.index import load_index  # noqa: E402
try:  # Support both `python -m topology.pin_locator` and direct execution.
    from .line_detector import generate_pin_candidates, refine_bbox_from_outline  # type: ignore
    from .pin_tip_detector import (  # type: ignore
        detect_local_terminal_candidates,
        refine_pin_candidates_with_wires,
    )
    from .wire_detector import detect_wire_segments  # type: ignore
except ImportError:
    from topology.line_detector import generate_pin_candidates, refine_bbox_from_outline  # type: ignore
    from topology.pin_tip_detector import (  # type: ignore
        detect_local_terminal_candidates,
        refine_pin_candidates_with_wires,
    )
    from topology.wire_detector import detect_wire_segments  # type: ignore


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def normalize_pin_name(value: Any) -> str:
    """Normalize harmless OCR separators for pin-name comparison only."""
    return re.sub(r"[\s_\-]+", "", str(value or "")).casefold()


def pin_sort_key(number: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(number))
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def _text_center(item: dict[str, Any]) -> tuple[float, float] | None:
    bbox = item.get("bbox_xyxy_top_left") or item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _text_height(item: dict[str, Any]) -> float:
    bbox = item.get("bbox_xyxy_top_left") or item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return 12.0
    return max(1.0, abs(float(bbox[3]) - float(bbox[1])))


def nearby_texts(
    point: dict[str, float],
    ocr_items: list[dict[str, Any]],
    radius: float = 45.0,
) -> list[dict[str, Any]]:
    matches = []
    for item in ocr_items:
        center = _text_center(item)
        if center is None:
            continue
        distance = float(np.hypot(center[0] - point["x"], center[1] - point["y"]))
        if distance <= radius:
            matches.append({"item": item, "distance": distance})
    return sorted(matches, key=lambda item: item["distance"])


def candidate_texts(
    candidate: dict[str, Any],
    ocr_items: list[dict[str, Any]],
    *,
    radius: float = 55.0,
    outward_only: bool = False,
) -> list[dict[str, Any]]:
    """Return OCR items close to one side-local pin candidate."""
    point = candidate["point_top_left"]
    matches: list[dict[str, Any]] = []
    for item in ocr_items:
        center = _text_center(item)
        if center is None:
            continue
        dx = center[0] - point["x"]
        dy = center[1] - point["y"]
        distance = float(np.hypot(dx, dy))
        # Pin rows/columns must align. This rejects text from an adjacent row
        # even when it falls inside the old global 45 px circle.
        if candidate.get("side") in {"left", "right"}:
            aligned = abs(dy) <= max(12.0, 1.4 * _text_height(item) + 5.0)
        else:
            aligned = abs(dx) <= max(12.0, 1.4 * _text_height(item) + 5.0)
        if outward_only:
            if candidate.get("side") == "left":
                aligned = aligned and center[0] <= point["x"] + 3.0
            elif candidate.get("side") == "right":
                aligned = aligned and center[0] >= point["x"] - 3.0
            elif candidate.get("side") == "top":
                aligned = aligned and center[1] <= point["y"] + 3.0
            elif candidate.get("side") == "bottom":
                aligned = aligned and center[1] >= point["y"] - 3.0
        if distance > radius or not aligned:
            continue
        matches.append(
            {
                "item": item,
                "distance": distance,
                "axis_error": abs(dy) if candidate.get("side") in {"left", "right"} else abs(dx),
            }
        )
    return sorted(matches, key=lambda item: (item["axis_error"], item["distance"]))


PIN_NUMBER_RE = re.compile(r"^[A-Za-z]{0,3}\d+(?:_\d+)?$")
OCR_NUMBER_MIN_SCORE = 0.30
OCR_NAME_MIN_SCORE = 0.45


def compact_pin_number(value: Any) -> str:
    text = normalize_text(value).strip(".,:;()[]{}")
    return text.replace("－", "-").replace(" ", "")


def base_pin_number(value: str, known_numbers: set[str]) -> str:
    """Strip a converter duplicate suffix only when the base is known."""
    if value in known_numbers:
        return value
    match = re.fullmatch(r"(.+)_\d+", value)
    if match and match.group(1) in known_numbers:
        return match.group(1)
    return value


def number_evidence(
    candidate: dict[str, Any],
    ocr_items: list[dict[str, Any]],
    allowed_numbers: set[str],
) -> tuple[str | None, float, dict[str, Any] | None]:
    matches: list[tuple[str, float, dict[str, Any]]] = []
    for match in candidate_texts(candidate, ocr_items, outward_only=True):
        text_center = _text_center(match["item"])
        if text_center is not None:
            axis_coordinate = text_center[1] if candidate.get("side") in {"left", "right"} else text_center[0]
            point_coordinate = (
                float(candidate["point_top_left"]["y"])
                if candidate.get("side") in {"left", "right"}
                else float(candidate["point_top_left"]["x"])
            )
            # In the contest schematics the printed pin number is normally
            # inward/upstream of the terminal. Reject the next row's number
            # on the far side of the candidate; without this, pin 2 can be
            # assigned to pin 1 because the two OCR boxes overlap the search
            # radius.
            if axis_coordinate > point_coordinate + 3.0:
                continue
        compact = compact_pin_number(match["item"].get("text", ""))
        if compact not in allowed_numbers and not re.fullmatch(r"\d+(?:_\d+)?", compact):
            continue
        number = base_pin_number(compact, allowed_numbers)
        if number not in allowed_numbers and not PIN_NUMBER_RE.fullmatch(number):
            continue
        # Schematic pin numbers are often printed one text height inward from
        # the actual terminal, so allow a roughly 35 px row offset while
        # retaining side alignment as the main safeguard.
        score = max(0.0, 1.0 - match["axis_error"] / 35.0) * max(
            0.1, float(match["item"].get("score", 0.0) or 0.0)
        )
        matches.append((number, score, match["item"]))
    if not matches:
        return None, 0.0, None
    number, score, item = max(matches, key=lambda value: value[1])
    return number, score, item


def name_evidence(
    candidate: dict[str, Any],
    ocr_items: list[dict[str, Any]],
    pin_table: dict[str, list[dict[str, str]]],
) -> tuple[str | None, float, dict[str, Any] | None]:
    normalized_names: dict[str, set[str]] = {}
    for number, alternatives in pin_table.items():
        for alternative in alternatives:
            name = normalize_pin_name(alternative.get("name", ""))
            if name and name != normalize_pin_name(number):
                normalized_names.setdefault(name, set()).add(str(number))
    matches: list[tuple[str, float, dict[str, Any]]] = []
    for match in candidate_texts(candidate, ocr_items):
        text = normalize_pin_name(match["item"].get("text", ""))
        number_set = normalized_names.get(text, set())
        if len(number_set) != 1:
            continue
        number = next(iter(number_set))
        score = max(0.0, 1.0 - match["axis_error"] / 25.0) * max(
            0.1, float(match["item"].get("score", 0.0) or 0.0)
        )
        matches.append((number, score, match["item"]))
    if not matches:
        return None, 0.0, None
    return max(matches, key=lambda value: value[1])


def visible_pin_name_evidence(
    candidate: dict[str, Any],
    ocr_items: list[dict[str, Any]],
) -> tuple[str | None, float, dict[str, Any] | None]:
    """Find a visible, non-numeric label without using the IC library.

    This is deliberately only evidence.  Without a verified library record a
    pin name cannot determine the protocol pin key, so it is never converted
    into a fabricated ``pin_<number>`` entry by itself.
    """
    matches: list[tuple[str, float, dict[str, Any]]] = []
    for match in candidate_texts(candidate, ocr_items):
        text = str(match["item"].get("text", "")).strip()
        compact = compact_pin_number(text)
        if not text or PIN_NUMBER_RE.fullmatch(compact):
            continue
        if re.fullmatch(r"(?:[A-Za-z]{1,4})\d+", compact):
            continue
        if len(compact) > 48 or not re.search(r"[A-Za-z]", compact):
            continue
        score = max(0.0, 1.0 - match["axis_error"] / 25.0) * max(
            0.1, float(match["item"].get("score", 0.0) or 0.0)
        )
        matches.append((text, score, match["item"]))
    if not matches:
        return None, 0.0, None
    return max(matches, key=lambda value: value[1])


def _candidate_axis(candidate: dict[str, Any]) -> float:
    point = candidate["point_top_left"]
    return float(point["y"] if candidate.get("side") in {"left", "right"} else point["x"])


def _text_axis(item: dict[str, Any], side: str) -> float | None:
    center = _text_center(item)
    if center is None:
        return None
    return float(center[1] if side in {"left", "right"} else center[0])


def _is_inward_text(
    candidate: dict[str, Any],
    center: tuple[float, float],
    bbox_top_left: list[float],
) -> bool:
    """Require a pin number/name to be on the component-facing side.

    Pin labels in the public drawings are normally printed inside the symbol,
    whereas net labels and neighbouring component IDs are outside the tip.
    A small 4 px slack keeps anti-aliased text boxes from being rejected.
    """
    x1, y1, x2, y2 = map(float, bbox_top_left)
    x, y = center
    side = candidate.get("side")
    if side == "left":
        return x >= x1 - 4.0
    if side == "right":
        return x <= x2 + 4.0
    if side == "top":
        return y >= y1 - 4.0
    if side == "bottom":
        return y <= y2 + 4.0
    return False


def _side_compatible_text(
    candidate: dict[str, Any],
    center: tuple[float, float],
    bbox_top_left: list[float],
) -> bool:
    """Require interior OCR to remain on the candidate's half of an IC.

    ``_is_inward_text`` rejects exterior net labels, but by itself a right-side
    pin name can still be paired with a left-side terminal because both labels
    lie inside a rectangular IC.  The half-plane test is scale-relative and
    deliberately conservative: a label on the center line is ambiguous and
    is left unresolved rather than crossing the component.
    """
    x1, y1, x2, y2 = map(float, bbox_top_left)
    x, y = center
    slack_x = max(2.0, 0.08 * (x2 - x1))
    slack_y = max(2.0, 0.08 * (y2 - y1))
    side = candidate.get("side")
    if side == "left":
        return x <= (x1 + x2) * 0.5 + slack_x
    if side == "right":
        return x >= (x1 + x2) * 0.5 - slack_x
    if side == "top":
        return y <= (y1 + y2) * 0.5 + slack_y
    if side == "bottom":
        return y >= (y1 + y2) * 0.5 - slack_y
    return False


def _pin_label_records(
    ocr_items: list[dict[str, Any]],
    pin_table: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Convert OCR text into possible explicit pin identities.

    A library maps an observed *name* to a known pin identity, but never
    supplies an identity that was absent from OCR.  Without a library, a clear
    non-numeric pin label can still be used as a textual identity (e.g. VBUS).
    """
    known_numbers = set(pin_table)
    # A pin name is not a pin number when the library reuses that name for
    # several physical pins (for example NC).  Keeping only one number here
    # silently turns an ambiguous OCR label into a library-order guess.
    name_to_numbers: dict[str, set[str]] = {}
    for number, alternatives in pin_table.items():
        for alternative in alternatives:
            name = normalize_pin_name(alternative.get("name", ""))
            if name:
                name_to_numbers.setdefault(name, set()).add(str(number))
    records: list[dict[str, Any]] = []
    for text_index, item in enumerate(ocr_items):
        center = _text_center(item)
        if center is None:
            continue
        raw = str(item.get("text", "")).strip()
        compact = compact_pin_number(raw)
        if not compact:
            continue
        normalized_name = normalize_pin_name(raw)
        source: str | None = None
        identity: str | None = None
        # A pure numeral is unambiguously a printed pin number.  For mixed
        # tokens such as V3, DP2 or SBU1, prefer a unique visible pin-name
        # mapping before treating the token as a number: otherwise OCR pin
        # names become invented keys like ``pin_v3`` even when the library
        # explicitly identifies V3 as pin 4.  This is semantic validation,
        # not library-order completion.
        if compact in known_numbers or re.fullmatch(r"\d+(?:_\d+)?", compact):
            identity = base_pin_number(compact, known_numbers)
            source = "ocr_pin_number"
        elif len(name_to_numbers.get(normalized_name, set())) == 1:
            identity = next(iter(name_to_numbers[normalized_name]))
            source = "ocr_pin_name"
        if identity is None and not known_numbers:
            # Preserve only concise, visible alphabetic labels.  Network
            # labels are not generated here; their final assignment still
            # requires side/axis evidence in the monotonic matcher.
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_#/+\-]{0,31}", compact):
                identity = raw
                source = "ocr_pin_name"
        if identity is None or source is None:
            continue
        records.append(
            {
                "text_index": text_index,
                "item": item,
                "identity": str(identity),
                "source": source,
                "ocr_score": float(item.get("score", 0.0) or 0.0),
                "center": center,
            }
        )
    return records


def _monotonic_side_match(
    candidate_indices: list[int],
    candidates: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    bbox_top_left: list[float],
) -> dict[int, dict[str, Any]]:
    """Maximum-score non-crossing assignment for one IC side.

    Dynamic programming is used rather than nearest-neighbour matching: pins
    and their text are ordered along the same side, so crossed assignments are
    physically implausible and destabilise dense 10 px-pitch ICs.
    """
    if not candidate_indices or not labels:
        return {}
    side = candidates[candidate_indices[0]].get("side")
    ordered_candidates = sorted(candidate_indices, key=lambda index: _candidate_axis(candidates[index]))
    ordered_labels = sorted(labels, key=lambda item: _text_axis(item["item"], str(side)) or -1e9)
    rows, columns = len(ordered_candidates), len(ordered_labels)
    scores = np.zeros((rows, columns), dtype=np.float64)
    for row, candidate_index in enumerate(ordered_candidates):
        candidate = candidates[candidate_index]
        point = candidate["point_top_left"]
        for column, label in enumerate(ordered_labels):
            center = label["center"]
            axis_error = abs(_candidate_axis(candidate) - _text_axis(label["item"], str(side)))
            inward = _is_inward_text(candidate, center, bbox_top_left)
            number_outside_but_local = (
                label.get("source") == "ocr_pin_number"
                and axis_error <= max(8.0, 1.2 * _text_height(label["item"]) + 3.0)
                and float(np.hypot(center[0] - point["x"], center[1] - point["y"])) <= 28.0
            )
            # Numeric labels are often printed immediately outside an IC body.
            # They are valid only as local OCR evidence; this exception never
            # assigns a number from library order.
            if not inward and not number_outside_but_local:
                continue
            if inward and not _side_compatible_text(candidate, center, bbox_top_left):
                continue
            # Text is often printed inside the body, hence use axis alignment
            # as the decisive criterion and only a weak radial penalty.
            if axis_error > max(12.0, 1.4 * _text_height(label["item"]) + 5.0):
                continue
            radial = float(np.hypot(center[0] - point["x"], center[1] - point["y"]))
            if radial > 70.0:
                continue
            scores[row, column] = max(
                0.0,
                0.60 * float(label["ocr_score"])
                + 0.35 * (1.0 - min(1.0, axis_error / 18.0))
                + 0.05 * max(0.0, 1.0 - radial / 70.0),
            )
    # Sequence-alignment DP: skips cost zero; a positive match advances both
    # sequences and consequently cannot cross another selected match.
    dp = np.zeros((rows + 1, columns + 1), dtype=np.float64)
    decision = np.zeros((rows + 1, columns + 1), dtype=np.int8)
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            choices = (dp[row - 1, column], dp[row, column - 1], dp[row - 1, column - 1] + scores[row - 1, column - 1])
            best = int(np.argmax(choices))
            dp[row, column] = choices[best]
            decision[row, column] = best
    output: dict[int, dict[str, Any]] = {}
    row, column = rows, columns
    while row > 0 and column > 0:
        choice = int(decision[row, column])
        if choice == 2 and scores[row - 1, column - 1] >= OCR_NUMBER_MIN_SCORE:
            candidate_index = ordered_candidates[row - 1]
            label = dict(ordered_labels[column - 1])
            label["score"] = round(float(scores[row - 1, column - 1]), 4)
            output[candidate_index] = label
            row -= 1
            column -= 1
        elif choice == 0:
            row -= 1
        else:
            column -= 1
    return output


def assign_pins_monotonically(
    candidates: list[dict[str, Any]],
    ocr_items: list[dict[str, Any]],
    pin_table: dict[str, list[dict[str, str]]],
    bbox_top_left: list[float],
) -> dict[int, dict[str, Any]]:
    """Bind explicit OCR pin evidence while preserving per-side order."""
    labels = _pin_label_records(ocr_items, pin_table)
    assignments: dict[int, dict[str, Any]] = {}
    for side in ("left", "top", "right", "bottom"):
        indices = [index for index, item in enumerate(candidates) if item.get("side") == side]
        assignments.update(_monotonic_side_match(indices, candidates, labels, bbox_top_left))
    return assignments


def choose_pin_number(
    point: dict[str, float],
    ocr_items: list[dict[str, Any]],
    pin_numbers: set[str],
    used: set[str],
) -> str | None:
    # Backwards-compatible helper retained for callers outside this stage.
    for match in nearby_texts(point, ocr_items):
        text = str(match["item"].get("text", "")).strip()
        compact = normalize_text(text).strip(".,:;()[]")
        for number in sorted(pin_numbers - used, key=pin_sort_key):
            if compact == normalize_text(number):
                return number
    return None


def choose_pin_name(
    number: str,
    candidate: dict[str, Any],
    ocr_items: list[dict[str, Any]],
    pin_table: dict[str, list[dict[str, str]]],
) -> str:
    alternatives = pin_table.get(number, [])
    name_by_normalized = {
        normalize_pin_name(item.get("name")): item.get("name", "")
        for item in alternatives
        if item.get("name")
    }
    for match in candidate_texts(candidate, ocr_items):
        text = str(match["item"].get("text", "")).strip()
        normalized = normalize_pin_name(text)
        if normalized in name_by_normalized:
            return name_by_normalized[normalized]
    # Preserve only a sufficiently clear nearby non-numeric OCR label.  The
    # library is not allowed to fill a name that was not visibly supported by
    # OCR; it may only validate an observed label above.
    visible_name, visible_score, _ = visible_pin_name_evidence(candidate, ocr_items)
    if visible_name and visible_score >= OCR_NAME_MIN_SCORE:
        return visible_name
    return ""


def library_consistency(
    candidates: list[dict[str, Any]],
    pin_numbers: set[str],
    pin_table: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """Report image evidence versus the selected library pin table.

    A mismatch is diagnostic only.  It must not cause the whole IC to be
    dropped, because the contest specification does not make library lookup a
    prerequisite for emitting image-supported pin results.
    """
    visible_numbers = sorted(
        {
            str(item["number_evidence"])
            for item in candidates
            if item.get("number_evidence")
            and float(item.get("number_score", 0.0)) >= OCR_NUMBER_MIN_SCORE
        },
        key=pin_sort_key,
    )
    outside = (
        sorted(set(visible_numbers) - pin_numbers, key=pin_sort_key)
        if pin_numbers
        else []
    )
    library_names = {
        normalize_pin_name(item.get("name", ""))
        for alternatives in pin_table.values()
        for item in alternatives
        if item.get("name") and normalize_pin_name(item.get("name")) not in {
            normalize_pin_name(number) for number in pin_numbers
        }
    }
    visible_names = sorted(
        {
            str(item["visible_name_evidence"])
            for item in candidates
            if item.get("visible_name_evidence")
            and float(item.get("visible_name_score", 0.0)) >= OCR_NAME_MIN_SCORE
        },
        key=str.casefold,
    )
    names_outside = [
        name for name in visible_names if normalize_pin_name(name) not in library_names
    ]
    expected = len(pin_numbers)
    detected = len(candidates)
    if not pin_numbers:
        status = "unresolved"
    elif pin_numbers and (outside or names_outside):
        status = "inconsistent"
    elif detected == expected:
        status = "consistent"
    elif detected < expected:
        status = "partial_geometry"
    else:
        status = "inconsistent"
    return {
        "status": status,
        "expected_pin_count": expected or None,
        "detected_candidate_count": detected,
        "visible_ocr_number_count": len(visible_numbers),
        "visible_ocr_numbers": visible_numbers,
        "visible_numbers_outside_library": outside,
        "visible_ocr_names": visible_names,
        "visible_names_outside_library": names_outside if pin_numbers else [],
    }


def perimeter_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    point = candidate["point_top_left"]
    side = candidate["side"]
    if side == "left":
        return (0, point["y"], point["x"])
    if side == "top":
        return (1, point["x"], point["y"])
    if side == "right":
        return (2, point["y"], point["x"])
    return (3, point["x"], point["y"])


def select_candidates(
    raw_candidates: list[dict[str, Any]],
    pin_table: dict[str, list[dict[str, str]]],
    ocr_items: list[dict[str, Any]],
    *,
    allow_library_prune: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """Keep high-quality physical terminal candidates without global ordering."""
    expected_count = len(pin_table)
    annotated: list[dict[str, Any]] = []
    allowed_numbers = set(pin_table)
    for candidate in raw_candidates:
        number, number_score, _ = number_evidence(candidate, ocr_items, allowed_numbers)
        name, name_score, _ = name_evidence(candidate, ocr_items, pin_table)
        visible_name, visible_name_score, _ = visible_pin_name_evidence(candidate, ocr_items)
        item = dict(candidate)
        item["number_evidence"] = number
        item["number_score"] = round(number_score, 4)
        item["name_evidence"] = name
        item["name_score"] = round(name_score, 4)
        item["visible_name_evidence"] = visible_name
        item["visible_name_score"] = round(visible_name_score, 4)
        item["selection_score"] = round(
            0.55 * float(candidate.get("evidence_score", 0.0))
            + 0.30 * number_score
            + 0.10 * name_score
            + 0.05 * visible_name_score,
            4,
        )
        annotated.append(item)
    # Hough and raster scans can produce several rows for one thick pin. Work
    # side-by-side and collapse nearby candidates before applying the library
    # pin count. This is essential for the 5 px point threshold.
    grouped: list[dict[str, Any]] = []
    for side in ("left", "top", "right", "bottom"):
        side_items = sorted(
            [item for item in annotated if item.get("side") == side],
            key=lambda item: (
                item["point_top_left"]["y"]
                if side in {"left", "right"}
                else item["point_top_left"]["x"]
            ),
        )
        groups: list[list[dict[str, Any]]] = []
        for item in side_items:
            coordinate = (
                item["point_top_left"]["y"]
                if side in {"left", "right"}
                else item["point_top_left"]["x"]
            )
            if not groups:
                groups.append([item])
                continue
            previous = groups[-1][-1]
            previous_coordinate = (
                previous["point_top_left"]["y"]
                if side in {"left", "right"}
                else previous["point_top_left"]["x"]
            )
            # The current local ROI detector has already clustered raster
            # thickness.  Its valid pins can be only 8--10 px apart, so the
            # old 12 px legacy-Hough merge destroyed real adjacent terminals.
            # Retain those local candidates independently; apply the wider
            # merge only to compatibility candidates from the old detector.
            local_pair = (
                item.get("source") == "local_roi_bfs_terminal_path"
                and previous.get("source") == "local_roi_bfs_terminal_path"
            )
            merge_distance = 2.5 if local_pair else 12.0
            if coordinate - previous_coordinate <= merge_distance:
                groups[-1].append(item)
            else:
                groups.append([item])
        for group in groups:
            grouped.append(
                max(
                    group,
                    key=lambda item: (
                        # Prefer a line that reaches the component edge over
                        # an OCR glyph that happens to sit on the same row.
                        # OCR remains a label-assignment signal below.
                        float(item.get("evidence_score", 0.0)),
                        float(item.get("selection_score", 0.0)),
                        max(float(item.get("number_score", 0.0)), float(item.get("name_score", 0.0))),
                    ),
                )
            )
    # The same OCR label can be captured once by the scan and once by Hough
    # at slightly different offsets. Keep one candidate for a repeated
    # side-local number/name; duplicate physical numbers are handled only
    # when they occur on distinct sides or when the library explicitly
    # provides duplicate slots.
    deduplicated: list[dict[str, Any]] = []
    seen_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for item in sorted(
        grouped,
        key=lambda value: (
            -float(value.get("evidence_score", 0.0)),
            -float(value.get("selection_score", 0.0)),
            perimeter_key(value),
        ),
    ):
        number_score = float(item.get("number_score", 0.0))
        name_score = float(item.get("name_score", 0.0))
        if item.get("source") == "local_roi_bfs_terminal_path":
            # Local BFS has already produced one candidate per measured
            # terminal.  Two adjacent pins may legitimately share the same
            # OCR evidence or have no OCR at all; identity de-duplication here
            # would delete physical pins before the monotonic matcher runs.
            deduplicated.append(item)
            continue
        if max(number_score, name_score) < 0.30:
            identity = None
        elif name_score > number_score:
            identity = item.get("name_evidence")
        else:
            identity = item.get("number_evidence")
        if identity:
            key = (str(item.get("side")), str(identity))
            if key in seen_identity:
                continue
            seen_identity[key] = item
        deduplicated.append(item)
    annotated = deduplicated
    annotated = sorted(annotated, key=perimeter_key)
    if expected_count and len(annotated) > expected_count and allow_library_prune:
        # A library record is a semantic constraint, not permission to erase
        # image evidence.  If OCR visibly supports more terminals than the
        # selected record (for example a stale five-pin record beside a
        # six-terminal drawing), retain all measured candidates and report
        # the mismatch.  Downstream pin-number binding may still leave some
        # candidates unresolved, but their real geometry is not discarded.
        visible_evidence = [
            item
            for item in annotated
            if float(item.get("number_score", 0.0)) >= OCR_NUMBER_MIN_SCORE
            or max(
                float(item.get("name_score", 0.0)),
                float(item.get("visible_name_score", 0.0)),
            ) >= OCR_NAME_MIN_SCORE
        ]
        if len(visible_evidence) > expected_count:
            return sorted(visible_evidence, key=perimeter_key), False
        strong = [
            item
            for item in annotated
            if (
                max(float(item.get("number_score", 0.0)), float(item.get("name_score", 0.0))) >= 0.30
                or float(item.get("evidence_score", 0.0)) >= 0.90
            )
        ]
        pool = strong if len(strong) >= expected_count else annotated
        annotated = sorted(
            pool,
            key=lambda item: (
                -float(item.get("evidence_score", 0.0)),
                -float(item["selection_score"]),
                perimeter_key(item),
            ),
        )[:expected_count]
        return sorted(annotated, key=perimeter_key), True
    return sorted(annotated, key=perimeter_key), False


def locate_ic_pins(
    image: np.ndarray,
    bbox_top_left: list[float],
    ic_record: dict[str, Any] | None,
    ocr_items: list[dict[str, Any]],
    *,
    image_height: float,
    expected_offset: float = 12.0,
    wire_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Locate one IC and return image-supported pins plus diagnostics.

    The library is an optional semantic constraint.  Geometry is always kept
    in diagnostics, and a final pin is emitted only when its number/name is
    supported by sufficiently clear local OCR evidence.  In particular, this
    function never assigns unreadable pins by library order.
    """
    refined_bbox = refine_bbox_from_outline(image, bbox_top_left)
    raw_candidates, local_geometry = detect_local_terminal_candidates(
        image,
        refined_bbox,
        ocr_items=ocr_items,
    )
    pin_tip_method = "local_roi_terminal_intersection"
    # Compatibility rescue for old callers only.  A bbox offset is never the
    # normal coordinate source once local terminal geometry is available.
    if not raw_candidates and wire_segments is not None:
        raw_candidates = generate_pin_candidates(
            image,
            refined_bbox,
            expected_offset=expected_offset,
        )
        # The legacy offset is retained only to generate side-local search
        # rows.  It is never used as the submitted point when wire geometry
        # is available; the point is replaced by measured terminal evidence.
        raw_candidates = refine_pin_candidates_with_wires(
            image,
            refined_bbox,
            raw_candidates,
            wire_segments,
        )
        pin_tip_method = "legacy_wire_rescue"
    diagnostics: dict[str, Any] = {
        "bbox_top_left": bbox_top_left,
        "bbox_used_top_left": refined_bbox,
        "local_geometry": local_geometry,
        "raw_candidate_count": len(raw_candidates),
        "candidate_count": len(raw_candidates),
        "candidates": raw_candidates,
        "assigned_count": 0,
        "unassigned_pin_numbers": [],
        "candidate_pruned": False,
        "pin_tip_method": pin_tip_method,
        "pin_tip_offset_px": None,
        "wire_segment_count": len(wire_segments) if wire_segments is not None else None,
        "warnings": [],
    }
    pin_table = ic_record.get("pins_by_number", {}) if ic_record else {}
    pin_numbers = set(pin_table)
    candidates, pruned = select_candidates(
        raw_candidates,
        pin_table,
        ocr_items,
        allow_library_prune=False,
    )
    diagnostics["candidates"] = candidates
    diagnostics["candidate_pruned"] = pruned
    diagnostics["candidate_count"] = len(candidates)
    diagnostics["library_name"] = ic_record.get("name") if ic_record else None
    diagnostics["library_available"] = bool(pin_table)
    diagnostics["library_consistency"] = library_consistency(candidates, pin_numbers, pin_table)
    if not ic_record:
        diagnostics["warnings"].append(
            "no IC library record; retaining geometry and emitting only OCR-identified pin numbers"
        )
    elif diagnostics["library_consistency"]["status"] in {"inconsistent", "partial_geometry"}:
        diagnostics["warnings"].append(
            "library pin table and visible geometry/OCR evidence are not fully consistent"
        )
    ordered_candidates = candidates
    assignments = assign_pins_monotonically(
        ordered_candidates,
        ocr_items,
        pin_table,
        refined_bbox,
    )

    # One physical IC pin number must produce one formal key.  OCR may see
    # both a number and its name, or duplicate the same number on adjacent
    # rows.  Keeping both would create invalid-looking ``pin_7_1`` and
    # ``pin_7_2`` entries.  Retain the strongest image-supported assignment;
    # the discarded geometric candidate remains available in diagnostics and
    # is never completed from the library order.
    best_by_identity: dict[str, tuple[int, dict[str, Any]]] = {}
    duplicate_assignment_diagnostics: list[dict[str, Any]] = []
    for candidate_index, assignment in assignments.items():
        identity = str(assignment["identity"])
        rank = (
            1 if assignment.get("source") == "ocr_pin_number" else 0,
            float(assignment.get("score", 0.0)),
            float(assignment.get("ocr_score", 0.0)),
        )
        previous = best_by_identity.get(identity)
        if previous is None:
            best_by_identity[identity] = (candidate_index, assignment)
            continue
        old_index, old_assignment = previous
        old_rank = (
            1 if old_assignment.get("source") == "ocr_pin_name" else 0,
            float(old_assignment.get("score", 0.0)),
            float(old_assignment.get("ocr_score", 0.0)),
        )
        if rank > old_rank:
            duplicate_assignment_diagnostics.append({"dropped": old_assignment, "kept": assignment})
            best_by_identity[identity] = (candidate_index, assignment)
        else:
            duplicate_assignment_diagnostics.append({"dropped": assignment, "kept": old_assignment})
    assignments = {index: assignment for index, assignment in best_by_identity.values()}

    pins: dict[str, Any] = {}
    assignment_diagnostics: list[dict[str, Any]] = []
    for index, assignment in sorted(assignments.items()):
        candidate = ordered_candidates[index]
        number = str(assignment["identity"])
        pin_key = f"pin_{number}"
        if pin_key in pins:
            # The one-to-one identity filter above should make this
            # unreachable. Keep a hard guard so a later OCR change can never
            # reintroduce unsupported ``pin_10_1`` style duplicates.
            duplicate_assignment_diagnostics.append(
                {"dropped": assignment, "kept": "existing_formal_pin", "reason": "duplicate_pin_key"}
            )
            continue
        point = top_left_point_to_bottom_left(
            candidate["point_top_left"],
            image_height,
            decimals=2,
        )
        pins[pin_key] = {
            "pinname": choose_pin_name(number, candidate, ocr_items, pin_table),
            "point": point,
        }
        assignment_diagnostics.append(
            {
                "pin_key": pin_key,
                "number": number,
                "source": assignment["source"],
                "score": round(float(assignment.get("score", 0.0)), 4),
                "text_index": assignment.get("text_index"),
                "side": candidate.get("side"),
                "point_top_left": candidate.get("point_top_left"),
            }
        )

    diagnostics["assigned_count"] = len(pins)
    diagnostics["geometry_only_candidate_count"] = max(0, len(candidates) - len(assignments))
    diagnostics["assignments"] = assignment_diagnostics
    diagnostics["duplicate_identity_assignments_dropped"] = duplicate_assignment_diagnostics
    diagnostics["library_order_fallback_used"] = False
    diagnostics["unassigned_pin_numbers"] = sorted(
        pin_numbers - {str(item["identity"]) for item in assignments.values()},
        key=pin_sort_key,
    )
    if pin_numbers and len(candidates) != len(pin_numbers):
        diagnostics["warnings"].append(
            f"candidate count {len(candidates)} differs from library pin count {len(pin_numbers)}"
        )
    return {"pins": pins, "diagnostics": diagnostics}


def locate_non_ic_pins(
    image: np.ndarray,
    bbox_top_left: list[float],
    component_type: str,
    ocr_items: list[dict[str, Any]],
    *,
    image_height: float,
) -> dict[str, Any]:
    """Locate pins of a non-IC symbol from measured terminal geometry.

    The contest format still requires a ``pins`` entry for ordinary symbols.
    These symbols normally have no reliable printed pin number, so the PDF's
    documented automatic numbering rule is used: ``pin_1``, ``pin_2``, ... in
    perimeter order.  This is geometry order, not an IC-library order and it
    never creates a coordinate without a measured terminal path.
    """
    # Small-symbol detector boxes often cover only the red body.  Expand the
    # local search boundary by a scale-derived amount so terminal strokes just
    # outside that body can be observed.  The output point remains the actual
    # measured red/green transition; this expansion is never a coordinate
    # offset and never invents a missing pin.
    x1, y1, x2, y2 = map(float, bbox_top_left)
    symbol_scale = max(1.0, min(x2 - x1, y2 - y1))
    search_expand = max(1.0, min(4.0, round(0.25 * symbol_scale)))
    search_bbox = [x1 - search_expand, y1 - search_expand, x2 + search_expand, y2 + search_expand]
    raw_candidates, local_geometry = detect_local_terminal_candidates(
        image,
        search_bbox,
        ocr_items=ocr_items,
        minimum_reach=2,
    )
    candidates = sorted(raw_candidates, key=perimeter_key)
    pins: dict[str, Any] = {}
    for index, candidate in enumerate(candidates, start=1):
        point = top_left_point_to_bottom_left(
            candidate["point_top_left"], image_height, decimals=2
        )
        # R/C/switch symbols do not expose a trustworthy pin name in the
        # standard protocol.  Keep it as an empty string instead of copying a
        # nearby value/designator or inventing a semantic name.
        pins[f"pin_{index}"] = {"pinname": "", "point": point}
    return {
        "pins": pins,
        "diagnostics": {
            "component_type": component_type,
            "bbox_top_left": bbox_top_left,
            "search_bbox_top_left": search_bbox,
            "search_expand_px": search_expand,
            "raw_candidate_count": len(raw_candidates),
            "candidate_count": len(candidates),
            "pin_number_source": "contest_auto_number_from_measured_geometry",
            "candidates": candidates,
            "local_geometry": local_geometry,
            "warning": None if candidates else "no measured terminal path",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True, help="YOLO+OCR JSON")
    parser.add_argument("--component-json", type=Path, required=True, help="Component JSON")
    parser.add_argument(
        "--ic-name-map",
        type=Path,
        required=True,
        help='JSON: {"U1":"STM32..."} or resolver output with model_name fields',
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "ic_library" / "BoxICLibrary_200cases.index.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--with-global-wire-diagnostics",
        action="store_true",
        help="also run whole-page wire extraction for debugging; not needed for local pin tips",
    )
    parser.add_argument("--expected-offset", type=float, default=12.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def mapped_model_name(name_map: dict[str, Any], component_key: str) -> str | None:
    """Accept both the original simple map and resolver output."""
    value = name_map.get(component_key)
    if isinstance(value, dict):
        model_name = value.get("model_name") or value.get("name")
        return str(model_name).strip() if model_name else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main() -> None:
    args = parse_args()
    image = read_image(args.image)
    intermediate = json.loads(args.intermediate.read_text(encoding="utf-8"))
    component_payload = json.loads(args.component_json.read_text(encoding="utf-8"))
    name_map = json.loads(args.ic_name_map.read_text(encoding="utf-8"))
    index = load_index(args.index)
    image_height = float(image.shape[0])
    ocr_items = intermediate.get("ocr", {}).get("texts", [])
    all_components = component_payload.get("components", {})
    wire_segments: list[dict[str, Any]] | None = None
    if args.with_global_wire_diagnostics:
        component_boxes_top_left = [
            bottom_left_bbox_to_top_left(component["bbox"], image_height)
            for component in all_components.values()
            if isinstance(component, dict) and isinstance(component.get("bbox"), list)
        ]
        wire_segments = detect_wire_segments(
            image,
            component_boxes_top_left=component_boxes_top_left,
            ocr_items=ocr_items,
        )

    all_pins: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for component_key, component in component_payload.get("components", {}).items():
        if component.get("type") != "box":
            continue
        bbox_top_left = bottom_left_bbox_to_top_left(component["bbox"], image_height)
        model_name = mapped_model_name(name_map, component_key)
        record = index.get(model_name) if model_name else None
        result = locate_ic_pins(
            image,
            bbox_top_left,
            record,
            ocr_items,
            image_height=image_height,
            expected_offset=args.expected_offset,
            wire_segments=wire_segments,
        )
        if result["pins"]:
            all_pins[component_key] = result["pins"]
        diagnostics[component_key] = {
            "model_name": model_name,
            **result["diagnostics"],
        }

    output = {
        "coordinate_system": "bottom_left",
        "pins": all_pins,
        "wire_geometry_top_left": wire_segments or [],
        "diagnostics": diagnostics,
        "note": "This stage outputs pin data only; merge it with components before final JSON validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ICs={len(diagnostics)}, pin_components={len(all_pins)} -> {args.output}")


if __name__ == "__main__":
    main()
