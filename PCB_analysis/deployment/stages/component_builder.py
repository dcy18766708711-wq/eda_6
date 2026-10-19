#!/usr/bin/env python3
"""Assemble YOLO+OCR intermediate results into component JSON.

This stage intentionally stops before pin and net reconstruction. It creates a
protocol-valid component baseline and an optional debug sidecar containing OCR
role/candidate information needed by later stages.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.coordinate import top_left_bbox_to_bottom_left  # noqa: E402
from common.schema_validator import validate_submission  # noqa: E402
from common.type_mapper import load_type_mapping, map_component_type  # noqa: E402


DESIGNATOR_RE = re.compile(
    # Prefixes occurring repeatedly in the 200 public target files.  Rare
    # one-off noisy labels are deliberately not promoted to universal rules.
    r"^(?P<prefix>USB|LED|SW|TP|JP|CN|RV|VR|IC|R|C|L|U|Q|D|F|S|X|Y|J)"
    r"(?P<suffix>\d+[A-Z]?)$",
    re.IGNORECASE,
)
# H1/H2 is intentionally kept out of the generic designator parser.  In the
# public cases the same token can be a header designator, an IC pin name, or a
# net label.  It may become a component key only after the detector type and
# local geometry establish that it belongs to a header/other symbol.
HEADER_DESIGNATOR_RE = re.compile(r"^H(?P<suffix>\d+[A-Z]?)$", re.IGNORECASE)
PARTIAL_DESIGNATOR_RE = re.compile(
    r"^(?P<prefix>USB|LED|SW|TP|JP|CN|RV|VR|IC|R|C|L|U|Q|D|F|S|X|Y|J)\?$",
    re.IGNORECASE,
)
HEADER_PARTIAL_DESIGNATOR_RE = re.compile(r"^H\?$", re.IGNORECASE)
NET_LABELS = {
    "VCC",
    "VDD",
    "VSS",
    "GND",
    "SCL",
    "SDA",
    "MISO",
    "MOSI",
    "RESET",
    "NRST",
    "CLK",
    "IO",
}
VALUE_RE = re.compile(
    r"^[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?"
    r"(?:[kKmMgGuUnNpPμµΩRrFfVvAaHhWwSs%]+)?$"
)

# Keep value parsing independent of the source file's terminal encoding. This
# recognizes the units used by the schematics, including μF and Ω.
VALUE_RE = re.compile(
    r"^[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?\s*"
    r"(?:[kKmMgGuUnNpPμµΩohmOHMrFfVvAaHhWwSs%]+)?$"
)

VALUE_TYPES = {
    "r",
    "c",
    "l",
    "v",
    "d",
    "led",
    "fuse",
    "battary",
    "crystal",
}
PREFIX_BY_TYPE = {
    "box": "U",
    "r": "R",
    "c": "C",
    "l": "L",
    "d": "D",
    "led": "LED",
    "bjt": "Q",
    "mosfet": "Q",
    "switch": "SW",
    "fuse": "F",
    "testpoint": "TP",
    "gnd": "GND",
    "v": "V",
}

# The official Q&A says that power/ground components are scored by their
# coordinate/type rather than their unobservable internal "∅..." key.  These
# local IDs are therefore only stable references for subsequent local stages;
# they are deliberately not attempts to reconstruct an official key.
# These objects are evaluated by type/geometry rather than a visible
# reference-designator key.  M3 mounting holes are represented by coordinate
# keys in the public type list just like power/ground symbols.
COORDINATE_MATCHED_TYPES = {"gnd", "v", "m3螺丝"}


DESIGNATOR_TYPE_HINTS = {
    "R": "r",
    "C": "c",
    "L": "l",
    "U": "box",
    "IC": "box",
    "D": "d",
    "LED": "led",
    "SW": "switch",
    "S": "switch",
    "F": "fuse",
    "X": "crystal",
    "Y": "crystal",
}

TYPE_DESIGNATOR_PREFIXES = {
    "box": {"U", "IC", "USB", "JP", "CN", "J"},
    "amp": {"U", "IC"},
    "r": {"R"},
    "c": {"C"},
    "l": {"L"},
    "d": {"D"},
    "led": {"LED"},
    "switch": {"SW", "S"},
    "fuse": {"F"},
    "crystal": {"X", "Y"},
    "bjt": {"Q", "U"},
    "mosfet": {"Q", "U"},
    "testpoint": {"TP"},
    "pin": {"TP"},
    # H is not a type.  These two detector classes are the only contexts in
    # which an H<n> OCR token may be considered as a component designator.
    "circle_header": {"H"},
    "other": {"H"},
}


GLOBAL_ASSIGNMENT_MIN_SCORE = 0.48
GLOBAL_UNMATCHED_COST = 1.0 - GLOBAL_ASSIGNMENT_MIN_SCORE


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def canonical_designator(text: Any, *, allow_header: bool = False) -> str | None:
    """Return a stable key candidate from one OCR text item.

    The OCR box is allowed to contain spaces or a hyphen, but the submitted
    component key must remain the compact designator used by the GT files.
    """
    # Rebuild from the original OCR spelling, then canonicalize the prefix.
    # OCR often returns ``u9``/``r2`` even though component keys are
    # case-insensitive identifiers in the image and uppercase in the protocol.
    # Keeping the original case here would make ``u9`` and ``U9`` appear to be
    # different keys during Hungarian matching and duplicate checks.
    compact = normalize_text(text).replace("_", "").replace("-", "")
    match = DESIGNATOR_RE.fullmatch(compact)
    if not match and allow_header:
        header = HEADER_DESIGNATOR_RE.fullmatch(compact)
        if header:
            return f"H{(header.group('suffix') or '').upper()}"
    if not match:
        return None
    return f"{match.group('prefix').upper()}{(match.group('suffix') or '').upper()}"


def partial_designator(text: Any, *, allow_header: bool = False) -> str | None:
    """Return an observed but unusable designator such as ``U?``.

    A question mark is present in a number of input drawings.  It establishes
    the prefix only; it is not visual evidence for a numeric component key.
    The value is retained in diagnostics, never promoted to a submission key.
    """
    compact = normalize_text(text).upper().replace("_", "").replace("-", "")
    match = PARTIAL_DESIGNATOR_RE.fullmatch(compact)
    if not match and allow_header and HEADER_PARTIAL_DESIGNATOR_RE.fullmatch(compact):
        return "H?"
    if not match:
        return None
    return f"{match.group('prefix').upper()}?"


def _bbox(item: dict[str, Any]) -> list[float] | None:
    value = item.get("bbox_xyxy_top_left") or item.get("bbox")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(part) for part in value]
    except (TypeError, ValueError):
        return None


def _rectangle_distance(left: list[float], right: list[float]) -> float:
    dx = max(right[0] - left[2], left[0] - right[2], 0.0)
    dy = max(right[1] - left[3], left[1] - right[3], 0.0)
    return math.hypot(dx, dy)


def _bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def _axis_alignment_score(text_box: list[float], component_box: list[float]) -> float:
    """Score whether an OCR box is aligned with a component side.

    Reference designators are normally placed above/below a symbol or beside
    it.  This is intentionally a soft geometric feature: it does not assume a
    fixed orientation, pixel scale, font, or component size, and therefore is
    useful for both green and monochrome drawings without becoming a template
    rule.
    """
    tx1, ty1, tx2, ty2 = text_box
    cx1, cy1, cx2, cy2 = component_box
    tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    ccx, ccy = (cx1 + cx2) * 0.5, (cy1 + cy2) * 0.5
    tw, th = max(1.0, tx2 - tx1), max(1.0, ty2 - ty1)
    cw, ch = max(1.0, cx2 - cx1), max(1.0, cy2 - cy1)
    x_overlap = max(0.0, min(tx2, cx2) - max(tx1, cx1)) / max(1.0, min(tw, cw))
    y_overlap = max(0.0, min(ty2, cy2) - max(ty1, cy1)) / max(1.0, min(th, ch))
    x_center = max(0.0, 1.0 - abs(tcx - ccx) / max(1.0, 0.5 * (cw + tw)))
    y_center = max(0.0, 1.0 - abs(tcy - ccy) / max(1.0, 0.5 * (ch + th)))

    # Use the axis perpendicular to the nearest separated side.  If the OCR
    # box overlaps the component (common for tiny symbols), retain a neutral
    # value rather than rejecting it based on a false side assumption.
    gap_x = max(cx1 - tx2, tx1 - cx2, 0.0)
    gap_y = max(cy1 - ty2, ty1 - cy2, 0.0)
    if gap_y >= gap_x and gap_y > 0.0:
        return float(0.65 * x_center + 0.35 * x_overlap)
    if gap_x > 0.0:
        return float(0.65 * y_center + 0.35 * y_overlap)
    return 0.5


def active_detection_indices(detections: list[dict[str, Any]], iou_threshold: float = 0.88) -> list[int]:
    """Remove near-identical detections before assigning OCR text.

    The ONNX decoder uses class-aware NMS, so the same physical symbol can
    survive twice when two classes overlap. Only very high-overlap boxes are
    removed here; neighboring real components are left untouched.
    """
    order = sorted(
        range(len(detections)),
        key=lambda index: float(detections[index].get("confidence", 0.0) or 0.0),
        reverse=True,
    )
    kept: list[int] = []
    for index in order:
        box = detections[index].get("bbox_xyxy_top_left")
        if not isinstance(box, list) or len(box) != 4:
            continue
        if any(
            _bbox_iou(box, detections[other]["bbox_xyxy_top_left"]) >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(index)
    return sorted(kept)


def classify_text(text: str, component_type: str) -> str:
    """Classify OCR text without changing its original spelling."""
    normalized = normalize_text(text)
    if not normalized:
        return "empty"
    upper = normalized.upper()
    if _canonical_for_type(upper, component_type):
        return "designator"
    if _partial_for_type(upper, component_type):
        return "partial_designator"
    if upper in NET_LABELS or upper.startswith("VCC") or upper.startswith("GND"):
        return "net_label"
    if component_type in VALUE_TYPES and VALUE_RE.fullmatch(normalized):
        return "value"
    if component_type == "box":
        return "component_name_candidate"
    if any(char.isdigit() for char in normalized) and len(normalized) <= 6:
        return "value_candidate"
    return "nearby_text"


def load_intermediate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for field in ("image_size", "yolo", "ocr", "component_text_links"):
        if field not in payload:
            raise ValueError(f"intermediate JSON missing {field!r}: {path}")
    return payload


def candidate_indices_for_component(payload: dict[str, Any], index: int) -> list[int]:
    links = payload.get("component_text_links", [])
    if index >= len(links):
        return []
    candidates = links[index].get("text_candidates", [])
    return [int(item["text_index"]) for item in candidates if "text_index" in item]


def component_text_candidates(
    payload: dict[str, Any],
    detection_index: int,
    *,
    context_margin: float = 90.0,
    allow_header: bool = False,
) -> list[dict[str, Any]]:
    """Expand the linker output with nearby OCR text.

    The upstream linker keeps only eight items. Dense ICs often fill those
    slots with pin names and numbers, dropping the designator above the body.
    This second pass restores nearby designators/model text while preserving
    the linker's association score when available.
    """
    detections = payload.get("yolo", {}).get("detections", [])
    texts = payload.get("ocr", {}).get("texts", [])
    if detection_index >= len(detections):
        return []
    component_box = detections[detection_index].get("bbox_xyxy_top_left")
    if not isinstance(component_box, list) or len(component_box) != 4:
        return []
    width = max(1.0, float(component_box[2]) - float(component_box[0]))
    height = max(1.0, float(component_box[3]) - float(component_box[1]))
    max_distance = max(context_margin, min(180.0, 0.7 * max(width, height)))
    link_scores: dict[int, float] = {}
    links = payload.get("component_text_links", [])
    if detection_index < len(links):
        for item in links[detection_index].get("text_candidates", []):
            try:
                link_scores[int(item["text_index"])] = float(item.get("association_score", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

    candidates: list[dict[str, Any]] = []
    for text_index, item in enumerate(texts):
        if not isinstance(item, dict):
            continue
        text_box = _bbox(item)
        if text_box is None:
            continue
        distance = _rectangle_distance(text_box, [float(v) for v in component_box])
        linked_score = link_scores.get(text_index)
        if linked_score is None and distance > max_distance:
            continue
        spatial_score = max(0.0, 1.0 - distance / max(1.0, max_distance))
        association_score = max(float(linked_score or 0.0), spatial_score)
        text_box = [float(value) for value in text_box]
        direction_score = _axis_alignment_score(text_box, [float(v) for v in component_box])
        candidates.append(
            {
                "text_index": text_index,
                "text": item.get("text", ""),
                "score": float(item.get("score", 0.0) or 0.0),
                "association_score": round(association_score, 6),
                "distance_px": round(distance, 2),
                "direction_score": round(direction_score, 6),
                "linked": linked_score is not None,
                "source": item.get("source"),
                "crop_top_left": item.get("crop_top_left"),
                "ocr_scale": item.get("ocr_scale", item.get("upscale")),
                "ocr_rotation": item.get("ocr_rotation", item.get("rotation_degrees", 0)),
            }
        )
    candidates.sort(
        key=lambda item: (
            0
            if canonical_designator(item["text"], allow_header=allow_header)
            else 1
            if partial_designator(item["text"], allow_header=allow_header)
            else 2,
            -(0.65 * item["score"] + 0.35 * item["association_score"]),
            -item.get("direction_score", 0.0),
            item["distance_px"],
        )
    )
    return candidates


def is_likely_false_box(detection: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    """Reject a small box false positive when OCR identifies a nearby part."""
    box = detection.get("bbox_xyxy_top_left")
    if not isinstance(box, list) or len(box) != 4:
        return False
    width = max(0.0, float(box[2]) - float(box[0]))
    height = max(0.0, float(box[3]) - float(box[1]))
    nearby_part_designator = False
    nearby_model = False
    for item in candidates:
        designator = canonical_designator(item.get("text"))
        score = float(item.get("association_score", 0.0))
        distance = float(item.get("distance_px", 999.0))
        if designator:
            prefix = re.match(r"[A-Z]+", designator).group(0)
            if prefix not in {"U", "IC"} and score >= 0.70 and distance <= 25.0:
                nearby_part_designator = True
        compact = normalize_text(item.get("text"))
        if (
            len(compact) >= 4
            and any(char.isdigit() for char in compact)
            and not designator
            and score >= 0.65
            and distance <= 35.0
        ):
            nearby_model = True
    return nearby_part_designator and not nearby_model and width * height < 1200.0


def choose_designator(
    candidates: list[dict[str, Any]],
    allowed_prefixes: set[str] | None = None,
) -> tuple[str | None, dict[str, Any] | None]:
    designators = []
    for item in candidates:
        designator = canonical_designator(item.get("text"))
        if not designator:
            continue
        prefix_match = re.match(r"[A-Z]+", designator)
        if allowed_prefixes and (not prefix_match or prefix_match.group(0) not in allowed_prefixes):
            continue
        designators.append(item)
    if not designators:
        return None, None
    best = max(
        designators,
        key=lambda item: (
            0.65 * float(item.get("score", 0.0))
            + 0.35 * float(item.get("association_score", 0.0)),
            -float(item.get("distance_px", 0.0)),
        ),
    )
    return canonical_designator(best.get("text")), best


def designator_evidence_summary(
    candidates: list[dict[str, Any]],
    allowed_prefixes: set[str] | None,
    text_items: list[dict[str, Any]],
    *,
    limit: int = 6,
    allow_header: bool = False,
) -> list[dict[str, Any]]:
    """Return the visible key candidates without inventing a key.

    This is deliberately kept in the debug sidecar.  It makes a wrong OCR
    key distinguishable from an unresolved key during public-set regression,
    while the submission JSON still follows the rule that only explicit
    numeric OCR evidence may become ``Name``.
    """
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            text_index = int(candidate["text_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if text_index < 0 or text_index >= len(text_items):
            continue
        raw_text = text_items[text_index].get("text", "")
        designator = canonical_designator(raw_text, allow_header=allow_header)
        if not designator:
            continue
        prefix = _designator_prefix(designator)
        if allowed_prefixes and prefix not in allowed_prefixes:
            continue
        evidence_score = (
            0.65 * float(text_items[text_index].get("score", 0.0) or 0.0)
            + 0.35 * float(candidate.get("association_score", 0.0) or 0.0)
        )
        output.append(
            {
                "text_index": text_index,
                "text": str(raw_text),
                "canonical": designator,
                "score": round(float(text_items[text_index].get("score", 0.0) or 0.0), 6),
                "association_score": round(float(candidate.get("association_score", 0.0) or 0.0), 6),
                "distance_px": round(float(candidate.get("distance_px", 0.0) or 0.0), 2),
                "direction_score": round(float(candidate.get("direction_score", 0.0) or 0.0), 6),
                "evidence_score": round(evidence_score, 6),
                "source": text_items[text_index].get("source"),
                "bbox_top_left": _bbox(text_items[text_index]),
                "crop_top_left": text_items[text_index].get("crop_top_left"),
                "ocr_scale": text_items[text_index].get("ocr_scale", text_items[text_index].get("upscale")),
                "ocr_rotation": text_items[text_index].get("ocr_rotation", text_items[text_index].get("rotation_degrees", 0)),
            }
        )
    output.sort(key=lambda item: (-item["evidence_score"], item["distance_px"], item["text_index"]))
    return output[:limit]


def _designator_prefix(value: str) -> str | None:
    match = re.match(r"[A-Z]+", value.upper())
    return match.group(0) if match else None


def _allows_header_designator(component_type: str) -> bool:
    """Allow H<n> as a component key only in header-like contexts."""
    return component_type in {"circle_header", "other"}


def _canonical_for_type(text: Any, component_type: str) -> str | None:
    return canonical_designator(text, allow_header=_allows_header_designator(component_type))


def _partial_for_type(text: Any, component_type: str) -> str | None:
    return partial_designator(text, allow_header=_allows_header_designator(component_type))


def assign_designators_globally(
    payload: dict[str, Any],
    detection_indices: list[int],
    detector_types: dict[int, str],
    candidate_cache: dict[int, list[dict[str, Any]]],
) -> dict[int, tuple[str, dict[str, Any]]]:
    """One-to-one assignment between component detections and OCR keys.

    Rows are YOLO detections and columns are unique OCR text boxes.  Extra
    dummy columns make leaving a component unresolved cheaper than accepting
    weak or type-incompatible text.  A second key-level pass prevents two OCR
    boxes containing the same designator from creating duplicate components.
    """
    if not detection_indices:
        return {}
    text_items = payload.get("ocr", {}).get("texts", [])
    rows = list(detection_indices)

    def solve(
        solve_rows: list[int],
        *,
        local_only: bool,
        blocked_designators: set[str],
    ) -> list[tuple[float, int, str, dict[str, Any]]]:
        """Solve one OCR source tier, leaving dummy matches unresolved."""
        candidate_by_pair: dict[tuple[int, int], dict[str, Any]] = {}
        text_indices: set[int] = set()
        for detection_index in solve_rows:
            component_type = detector_types[detection_index]
            allowed = TYPE_DESIGNATOR_PREFIXES.get(component_type)
            if component_type in COORDINATE_MATCHED_TYPES or not allowed:
                continue
            for candidate in candidate_cache.get(detection_index, []):
                try:
                    text_index = int(candidate["text_index"])
                except (KeyError, TypeError, ValueError):
                    continue
                if text_index < 0 or text_index >= len(text_items):
                    continue
                item = text_items[text_index]
                is_local = str(item.get("source", "")).startswith("component_local_ocr")
                if is_local != local_only:
                    continue
                designator = _canonical_for_type(item.get("text"), component_type)
                if not designator or designator in blocked_designators:
                    continue
                prefix = _designator_prefix(designator)
                if allowed and prefix not in allowed:
                    continue
                candidate_by_pair[(detection_index, text_index)] = candidate
                text_indices.add(text_index)
        if not text_indices:
            return []

        columns = sorted(text_indices)
        column_lookup = {value: index for index, value in enumerate(columns)}
        # One private dummy column per detection guarantees an unresolved option.
        costs = np.full((len(solve_rows), len(columns) + len(solve_rows)), 1e6, dtype=np.float64)
        scores: dict[tuple[int, int], float] = {}
        for row, detection_index in enumerate(solve_rows):
            detection = payload["yolo"]["detections"][detection_index]
            box = detection.get("bbox_xyxy_top_left", [0, 0, 1, 1])
            scale = max(12.0, float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
            costs[row, len(columns) + row] = GLOBAL_UNMATCHED_COST
            for (candidate_detection, text_index), candidate in candidate_by_pair.items():
                if candidate_detection != detection_index:
                    continue
                ocr_score = float(text_items[text_index].get("score", 0.0) or 0.0)
                association = float(candidate.get("association_score", 0.0) or 0.0)
                distance = float(candidate.get("distance_px", 1e6) or 0.0)
                if distance > max(35.0, 0.75 * scale):
                    continue
                text_box = _bbox(text_items[text_index])
                if text_box is not None:
                    center = ((text_box[0] + text_box[2]) / 2.0, (text_box[1] + text_box[3]) / 2.0)
                    owned_by_other_box = False
                    for other_index in detection_indices:
                        if other_index == detection_index or detector_types.get(other_index) != "box":
                            continue
                        other_box = payload["yolo"]["detections"][other_index].get("bbox_xyxy_top_left")
                        if (
                            isinstance(other_box, list)
                            and len(other_box) == 4
                            and float(other_box[0]) <= center[0] <= float(other_box[2])
                            and float(other_box[1]) <= center[1] <= float(other_box[3])
                        ):
                            owned_by_other_box = True
                            break
                    if owned_by_other_box:
                        continue
                distance_score = max(0.0, 1.0 - distance / max(30.0, 1.5 * scale))
                direction_score = float(candidate.get("direction_score", 0.5) or 0.5)
                linked_bonus = 0.03 if candidate.get("linked") else 0.0
                score = min(
                    1.0,
                    0.42 * ocr_score
                    + 0.33 * association
                    + 0.20 * distance_score
                    + 0.05 * direction_score
                    + linked_bonus,
                )
                column = column_lookup[text_index]
                costs[row, column] = 1.0 - score
                scores[(row, column)] = score

        selected: list[tuple[float, int, str, dict[str, Any]]] = []
        assigned_rows, assigned_columns = linear_sum_assignment(costs)
        for row, column in zip(assigned_rows.tolist(), assigned_columns.tolist()):
            if column >= len(columns):
                continue
            score = scores.get((row, column), 0.0)
            if score < GLOBAL_ASSIGNMENT_MIN_SCORE:
                continue
            detection_index = solve_rows[row]
            text_index = columns[column]
            designator = _canonical_for_type(
                text_items[text_index].get("text"),
                detector_types[solve_rows[row]],
            )
            if not designator:
                continue
            evidence = dict(candidate_by_pair[(detection_index, text_index)])
            evidence.update(
                {
                    "text": text_items[text_index].get("text", ""),
                    "score": float(text_items[text_index].get("score", 0.0) or 0.0),
                    "direction_score": round(direction_score, 6),
                    "global_assignment_score": round(score, 6),
                    "assignment_method": (
                        "type_gated_hungarian_local_fallback"
                        if local_only
                        else "type_gated_hungarian_full_ocr"
                    ),
                }
            )
            selected.append((score, detection_index, designator, evidence))
        return selected

    # Full-page OCR is authoritative for keys it visibly resolves.  Local
    # crops are a fallback for rows left unresolved and can never take a key
    # already claimed by full-page OCR.
    output: dict[int, tuple[str, dict[str, Any]]] = {}
    used_designators: set[str] = set()
    full_selected = solve(rows, local_only=False, blocked_designators=set())
    for _score, detection_index, designator, evidence in sorted(
        full_selected, key=lambda item: (-item[0], item[1], item[2])
    ):
        if designator in used_designators or detection_index in output:
            continue
        used_designators.add(designator)
        output[detection_index] = (designator, evidence)

    fallback_rows = [index for index in rows if index not in output]
    local_selected = solve(fallback_rows, local_only=True, blocked_designators=used_designators)
    for _score, detection_index, designator, evidence in sorted(
        local_selected, key=lambda item: (-item[0], item[1], item[2])
    ):
        if designator in used_designators or detection_index in output:
            continue
        used_designators.add(designator)
        output[detection_index] = (designator, evidence)
    return output


def choose_value(
    text_items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    component_type: str,
) -> str | None:
    for candidate in candidates:
        index = int(candidate["text_index"])
        if index >= len(text_items):
            continue
        text = normalize_text(text_items[index].get("text"))
        if classify_text(text, component_type) == "value":
            return text_items[index].get("text", text).strip()
    return None


def generated_key(component_type: str, sequence: int) -> str:
    """Create a local reference only for coordinate-matched power symbols."""
    return f"{component_type}_{sequence:03d}"


def unique_key(preferred: str | None, component_type: str, sequence: int, used: set[str]) -> str | None:
    """Return a supported submission key, or ``None`` when it is unknowable.

    Do not manufacture U1/R1/C1 from detector order or spatial order.  The
    contest uses direct component-key matching, and its duplicate-designator
    convention requires a source ``componentID`` ordering that is absent from
    raster input.  A duplicate observed designator is therefore ambiguous too.
    """
    if preferred:
        key = normalize_text(preferred)
        if key and key not in used:
            used.add(key)
            return key
        return None
    if component_type in COORDINATE_MATCHED_TYPES:
        key = generated_key(component_type, sequence)
        while key in used:
            sequence += 1
            key = generated_key(component_type, sequence)
        used.add(key)
        return key
    return None


def looks_like_ic_model(text: Any) -> bool:
    """Conservatively identify OCR text that may be an IC model name."""
    value = normalize_text(text)
    if len(value) < 3 or not any(char.isdigit() for char in value):
        return False
    upper = value.upper()
    if canonical_designator(upper) or upper in NET_LABELS:
        return False
    if "/" in value or "_" in value:
        return False
    if VALUE_RE.fullmatch(value):
        return False
    return True


def choose_ic_name(
    text_items: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    """Choose a provisional IC model; the library resolver is authoritative."""
    model_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for candidate in candidates:
        try:
            index = int(candidate["text_index"])
        except (KeyError, TypeError, ValueError):
            continue
        if index < 0 or index >= len(text_items):
            continue
        item = text_items[index]
        text = str(item.get("text", "")).strip()
        if not looks_like_ic_model(text):
            continue
        score = (
            0.60 * float(item.get("score", 0.0) or 0.0)
            + 0.30 * float(candidate.get("association_score", 0.0) or 0.0)
            - 0.10 * min(1.0, float(candidate.get("distance_px", 0.0) or 0.0) / 180.0)
        )
        model_candidates.append((score, text, candidate))
    if not model_candidates:
        return None, None
    score, text, evidence = max(model_candidates, key=lambda value: value[0])
    return text, {
        "text_index": evidence.get("text_index"),
        "text": text,
        "score": round(score, 6),
    }


def build_components(
    payload: dict[str, Any],
    *,
    type_mapping: dict[str, str] | None = None,
    strict_types: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_height = payload["image_size"].get("height")
    if not isinstance(image_height, (int, float)) or image_height <= 0:
        raise ValueError("intermediate JSON must contain a positive image_size.height")

    detections = payload["yolo"].get("detections", [])
    text_items = payload["ocr"].get("texts", [])
    components: dict[str, Any] = {}
    debug_components: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    type_counts: defaultdict[str, int] = defaultdict(int)

    active_indices = active_detection_indices(detections)
    detector_types: dict[int, str] = {}
    candidate_cache: dict[int, list[dict[str, Any]]] = {}
    false_box_indices: set[int] = set()
    for index in active_indices:
        detection = detections[index]
        detector_type = map_component_type(
            detection.get("class_name", "other"),
            type_mapping,
            strict=strict_types,
        )
        detector_types[index] = detector_type
        candidates = component_text_candidates(
            payload,
            index,
            allow_header=_allows_header_designator(detector_type),
        )
        candidate_cache[index] = candidates
        if detector_type == "box" and is_likely_false_box(detection, candidates):
            false_box_indices.add(index)
    global_designators = assign_designators_globally(
        payload,
        [index for index in active_indices if index not in false_box_indices],
        detector_types,
        candidate_cache,
    )
    # API text review is an optional last-resort source for a visible
    # designator.  It can propose text only; the detector bbox and type remain
    # authoritative.  The overrides are populated by the explicit recovery
    # stage below and are never generated from component order.
    for raw_index, evidence in payload.get("_designator_overrides", {}).items():
        try:
            override_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if override_index not in detector_types or override_index in global_designators:
            continue
        if not isinstance(evidence, dict) or not evidence.get("designator"):
            continue
        global_designators[override_index] = (
            str(evidence["designator"]),
            dict(evidence),
        )

    for index in active_indices:
        detection = detections[index]
        raw_type = detection.get("class_name", "other")
        detector_type = detector_types[index]
        bbox_top_left = detection.get("bbox_xyxy_top_left")
        if not isinstance(bbox_top_left, list) or len(bbox_top_left) != 4:
            raise ValueError(f"detection {index} has invalid bbox")
        bbox = top_left_bbox_to_bottom_left(bbox_top_left, image_height)
        candidates = candidate_cache[index]
        if index in false_box_indices:
            debug_components.append(
                {
                    "component_index": index,
                    "skipped": True,
                    "skip_reason": "box_detection_conflicts_with_nearby_non_ic_designator",
                    "raw_type": raw_type,
                    "detector_type": detector_type,
                    "confidence": detection.get("confidence"),
                    "bbox_top_left": bbox_top_left,
                    "text_candidates": candidates,
                }
            )
            continue
        allowed_prefixes = TYPE_DESIGNATOR_PREFIXES.get(detector_type)
        assignment = global_designators.get(index)
        designator, designator_evidence = assignment if assignment else (None, None)
        key_candidates = designator_evidence_summary(
            candidates,
            allowed_prefixes,
            text_items,
            allow_header=_allows_header_designator(detector_type),
        )
        all_key_candidates = designator_evidence_summary(
            candidates,
            None,
            text_items,
            # Keep H<n> in the audit/recovery candidate list even when the
            # YOLO class is ``box``.  H is still not eligible for the normal
            # global assignment on a box; it is only evidence for the
            # explicit recovery stage, where the local crop is reviewed.
            allow_header=True,
        )
        type_conflicting_candidates = [
            item
            for item in all_key_candidates
            if allowed_prefixes and _designator_prefix(item["canonical"]) not in allowed_prefixes
        ]
        partial_evidence = next(
            (
                item
                for item in candidates
                if _partial_for_type(item.get("text"), detector_type)
                and (
                    not allowed_prefixes
                    or re.match(r"[A-Z]+", _partial_for_type(item.get("text"), detector_type) or "").group(0)
                    in allowed_prefixes
                )
            ),
            None,
        )
        component_type = detector_type
        prefix = re.match(r"[A-Z]+", designator).group(0) if designator else ""
        # A raw box detection is already the strongest evidence that this is
        # an IC. Nearby R/C/Q designators often belong to surrounding parts;
        # they must not turn the IC into a resistor or capacitor.
        if detector_type != "box" and prefix:
            component_type = DESIGNATOR_TYPE_HINTS.get(prefix, component_type)
        candidate_texts = []
        for candidate in candidates:
            text_index = int(candidate["text_index"])
            if text_index >= len(text_items):
                continue
            raw_text = text_items[text_index].get("text", "")
            candidate_texts.append(
                {
                    "text_index": text_index,
                    "text": raw_text,
                    "role": classify_text(normalize_text(raw_text), component_type),
                    "score": text_items[text_index].get("score"),
                    "association_score": candidate.get("association_score"),
                    "distance_px": candidate.get("distance_px"),
                    "direction_score": candidate.get("direction_score"),
                    "source": candidate.get("source"),
                    "crop_top_left": candidate.get("crop_top_left"),
                    "ocr_scale": candidate.get("ocr_scale"),
                    "ocr_rotation": candidate.get("ocr_rotation", 0),
                }
            )

        type_counts[component_type] += 1
        key = unique_key(designator, component_type, type_counts[component_type], used_keys)
        if key is None:
            if designator:
                skip_reason = "duplicate_explicit_designator_requires_source_component_id"
                key_status = "duplicate_explicit_designator"
            elif key_candidates:
                # A visible numeric designator exists, but it was rejected by
                # the one-to-one assignment because a stronger component
                # already owns that key.  This is a conflict, not missing OCR;
                # rerunning a local crop cannot legitimately invent a suffix.
                skip_reason = "conflicting_explicit_designator_requires_source_component_id"
                key_status = "conflicting_explicit_designator"
            else:
                skip_reason = "unresolved_designator_requires_numeric_text_evidence"
                key_status = "unresolved_no_numeric_designator"
            debug_components.append(
                {
                    "component_index": index,
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "key_status": key_status,
                    "designator_status": (
                        "duplicate_explicit_designator"
                        if designator
                        else "conflicting_explicit_designator"
                        if key_candidates
                        else "unresolved_partial_designator"
                        if partial_evidence
                        else "type_incompatible_explicit_designator"
                        if type_conflicting_candidates
                        else "unresolved_missing_numeric_designator"
                    ),
                    "raw_type": raw_type,
                    "detector_type": detector_type,
                    "type": component_type,
                    "confidence": detection.get("confidence"),
                    "bbox_top_left": bbox_top_left,
                    "bbox_bottom_left": bbox,
                    "designator_evidence": designator_evidence,
                    "key_candidates": key_candidates,
                    "all_key_candidates": all_key_candidates,
                    "type_conflicting_candidates": type_conflicting_candidates,
                    "partial_designator_evidence": partial_evidence,
                    "text_candidates": candidate_texts,
                }
            )
            continue
        value = choose_value(text_items, candidates, component_type)
        ic_name, ic_name_evidence = (None, None)
        if component_type == "box":
            ic_name, ic_name_evidence = choose_ic_name(text_items, candidates)
        evidence_source = str((designator_evidence or {}).get("source") or "")
        if evidence_source.startswith("component_local_ocr"):
            key_source = "ocr_local_crop"
        elif evidence_source.startswith("qwen_designator"):
            key_source = "api_designator_review"
        elif evidence_source:
            key_source = "ocr_full_page"
        elif component_type in COORDINATE_MATCHED_TYPES:
            key_source = "coordinate_type_local_id"
        else:
            key_source = "unknown"
        key_audit = {
            "key_source": key_source,
            "key_ocr_text": (designator_evidence or {}).get("text"),
            "key_ocr_confidence": (designator_evidence or {}).get("score"),
            "key_ocr_bbox_top_left": _bbox(text_items[int(designator_evidence["text_index"])])
            if designator_evidence
            and isinstance(designator_evidence.get("text_index"), (int, float))
            and 0 <= int(designator_evidence["text_index"]) < len(text_items)
            else None,
            "key_ocr_crop_top_left": (designator_evidence or {}).get("crop_top_left"),
            "key_ocr_rotation": (designator_evidence or {}).get("ocr_rotation", 0),
            "key_api_result": (designator_evidence or {}).get("api_result"),
        }
        components[key] = {
            # Official targets use the component name/designator here (for
            # example U1).  The matched IC model is kept in ic_resolution and
            # is used to obtain pin metadata, not written into Name.
            "Name": key,
            "type": component_type,
            "value": value,
            "bbox": bbox,
            **key_audit,
        }
        debug_components.append(
            {
                "component_index": index,
                "key": key,
                "raw_type": raw_type,
                "detector_type": detector_type,
                "type": component_type,
                "type_source": "ocr_designator_prefix" if component_type != detector_type else "yolo_type",
                "designator_evidence": designator_evidence,
                "key_candidates": key_candidates,
                "all_key_candidates": all_key_candidates,
                "type_conflicting_candidates": type_conflicting_candidates,
                "key_status": "resolved_explicit_designator",
                "designator_status": "explicit_numeric_ocr",
                "partial_designator_evidence": partial_evidence,
                "ic_name_evidence": ic_name_evidence,
                **key_audit,
                "confidence": detection.get("confidence"),
                "bbox_top_left": bbox_top_left,
                "bbox_bottom_left": bbox,
                "text_candidates": candidate_texts,
            }
        )

    # Every non-coordinate component detection is expected to have a visible
    # designator in the contest cases.  Unresolved records are therefore a
    # recovery queue, not an accepted final state.
    eligible_debug = [
        item
        for item in debug_components
        if isinstance(item, dict)
        and item.get("detector_type") not in COORDINATE_MATCHED_TYPES
        and item.get("skip_reason") != "box_detection_conflicts_with_nearby_non_ic_designator"
    ]
    recovery_queue = [
        {
            "component_index": item.get("component_index"),
            "task": "recover_visible_designator",
            "reason": item.get("skip_reason", "unresolved_designator"),
            "detector_type": item.get("detector_type"),
            "bbox_top_left": item.get("bbox_top_left"),
            "partial_designator": item.get("partial_designator_evidence"),
            "candidates": item.get("all_key_candidates", []),
        }
        for item in eligible_debug
        if item.get("skipped")
    ]
    resolved_with_evidence = sum(
        1
        for item in eligible_debug
        if not item.get("skipped")
        and (
            str(item.get("key_source", "")).startswith("ocr_")
            or str(item.get("key_source", "")) == "api_designator_review"
        )
    )
    designator_recovery = {
        "eligible_detections": len(eligible_debug),
        "resolved_with_traceable_ocr": resolved_with_evidence,
        "recovery_rate": round(resolved_with_evidence / len(eligible_debug), 6)
        if eligible_debug
        else 1.0,
        "unresolved_designator_count": len(recovery_queue),
        "target_unresolved_designator_count": 0,
    }

    result = {"components": components, "pins": {}, "nets": {}}
    debug = {
        "source_image": payload.get("image"),
        "image_size": payload["image_size"],
        "coordinate_system": "bottom_left_xyxy",
        "components": debug_components,
        "designator_recovery": designator_recovery,
        "designator_recovery_queue": recovery_queue,
        "notes": [
            "pins and nets are intentionally empty at this stage",
            "component_name_candidate is preserved only in this debug sidecar",
            "unresolved designators are pending visual recovery, not accepted final components",
        ],
    }
    return result, debug


def unresolved_detection_indices(debug: dict[str, Any]) -> list[int]:
    """Return real detections whose visible designator needs recovery.

    Contest cases contain a visible designator for every scored component;
    partial, type-conflicting, and one-to-one assignment failures are all
    recoverable OCR tasks unless the detection was explicitly rejected as a
    false box.
    """
    output: list[int] = []
    for item in debug.get("components", []):
        if not isinstance(item, dict):
            continue
        if not item.get("skipped"):
            continue
        if item.get("skip_reason") == "box_detection_conflicts_with_nearby_non_ic_designator":
            continue
        if item.get("detector_type") in COORDINATE_MATCHED_TYPES:
            continue
        try:
            output.append(int(item["component_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(set(output))


def local_ocr_detection_indices(
    payload: dict[str, Any],
    debug: dict[str, Any],
    *,
    all_small_symbols: bool = False,
) -> list[int]:
    indices = set(unresolved_detection_indices(debug))
    if all_small_symbols:
        # Rotated/small reference designators are the common failure mode for
        # R/C/SW labels.  This is an explicit opt-in because it costs more
        # CPU time than the unresolved-only pass.
        target_types = {"r", "c", "switch", "bjt", "led", "d", "crystal"}
        for index, detection in enumerate(payload.get("yolo", {}).get("detections", [])):
            if str(detection.get("class_name", "")).casefold() in target_types:
                indices.add(index)
    return sorted(indices)


def add_local_ocr_evidence(
    payload: dict[str, Any],
    detection_indices: list[int],
    *,
    device: str,
    scales: list[float],
    rotations: list[int],
) -> dict[str, Any]:
    """Run optional multi-scale OCR only for unresolved component regions."""
    if not detection_indices:
        return {"requested": 0, "added": 0, "scales": scales, "rotations": rotations}
    image_path = Path(str(payload.get("image", "")))
    if not image_path.is_file():
        raise FileNotFoundError(f"local OCR image is unavailable: {image_path}")
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    from topology.ic_local_ocr import create_local_ocr, recognize_ic_crop
    from run_yolo_ocr import associate_texts

    ocr = create_local_ocr(device=device)
    detections = payload.get("yolo", {}).get("detections", [])
    existing = payload.setdefault("ocr", {}).setdefault("texts", [])
    added: list[dict[str, Any]] = []
    attempted = 0
    skipped_rotated = 0
    for detection_index in detection_indices:
        if detection_index < 0 or detection_index >= len(detections):
            continue
        bbox = detections[detection_index].get("bbox_xyxy_top_left")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        def consume(items: list[dict[str, Any]], scale: float, rotation: int) -> bool:
            """Merge local OCR and report whether this component got a key.

            A padded crop often contains a neighboring component's reference
            designator.  That text must not suppress the rotated recovery pass
            for the current detection.
            """
            nonlocal attempted
            saw_designator = False
            detection_box = [float(value) for value in bbox]
            detection_span = max(
                1.0,
                detection_box[2] - detection_box[0],
                detection_box[3] - detection_box[1],
            )
            local_designator_distance = max(24.0, 0.65 * detection_span)
            for item in items:
                item = dict(item)
                item["source"] = "component_local_ocr"
                item["component_index"] = detection_index
                item["ocr_scale"] = scale
                item["ocr_rotation"] = rotation
                box = _bbox(item)
                if box is not None:
                    text_distance = _rectangle_distance(box, detection_box)
                    if text_distance <= local_designator_distance and (
                        canonical_designator(item.get("text"))
                        or HEADER_DESIGNATOR_RE.fullmatch(normalize_text(item.get("text")))
                    ):
                        saw_designator = True
                duplicate = None
                for old in [*existing, *added]:
                    old_box = _bbox(old) if isinstance(old, dict) else None
                    if (
                        old_box is not None
                        and box is not None
                        and normalize_text(old.get("text")).casefold()
                        == normalize_text(item.get("text")).casefold()
                        and _bbox_iou(old_box, box) >= 0.60
                    ):
                        duplicate = old
                        break
                if duplicate is None:
                    # component_builder consumes the same canonical bbox field
                    # as full-page OCR.
                    item["bbox_xyxy_top_left"] = item.pop("bbox", box)
                    added.append(item)
                elif float(item.get("score", 0.0) or 0.0) > float(duplicate.get("score", 0.0) or 0.0):
                    duplicate["text"] = item.get("text", duplicate.get("text"))
                    duplicate["score"] = item.get("score", duplicate.get("score"))
            attempted += 1
            return saw_designator

        # Upright OCR is the cheapest and most reliable pass.  Rotated OCR is
        # only needed when all upright scales failed to expose a numeric key;
        # this avoids spending several PaddleOCR calls on already-resolved
        # R/C/Q symbols and was the main source of apparent pipeline hangs.
        zero_found = False
        for scale in scales:
            zero_found = consume(
                recognize_ic_crop(
                    ocr,
                    image,
                    bbox,
                    margin_ratio=0.35,
                    min_margin=32,
                    upscale=max(1.0, scale),
                    rotation_degrees=0,
                ),
                scale,
                0,
            ) or zero_found
        if not zero_found:
            for rotation in [value for value in rotations if value % 360 != 0]:
                for scale in scales:
                    consume(
                        recognize_ic_crop(
                            ocr,
                            image,
                            bbox,
                            margin_ratio=0.35,
                            min_margin=32,
                            upscale=max(1.0, scale),
                            rotation_degrees=rotation,
                        ),
                        scale,
                        rotation,
                    )
        else:
            skipped_rotated += max(0, len([value for value in rotations if value % 360 != 0])) * len(scales)
    existing.extend(added)
    payload["component_text_links"] = associate_texts(detections, existing)
    return {
        "requested": len(detection_indices),
        "added": len(added),
        "attempted_crops": attempted,
        "skipped_rotated_crops": skipped_rotated,
        "scales": scales,
        "rotations": rotations,
    }


def add_api_designator_evidence(
    payload: dict[str, Any],
    debug: dict[str, Any],
    *,
    enabled: bool,
) -> dict[str, Any]:
    """Use Qwen only to review unresolved visible designator text."""
    report: dict[str, Any] = {
        "enabled": bool(enabled),
        "requested": 0,
        "accepted": 0,
        "failed": 0,
        "results": [],
    }
    if not enabled:
        return report
    try:
        import api_config
        from topology.qwen_resolver import accept_qwen_designator, call_qwen_designator
    except Exception as exc:
        report["error"] = f"API module unavailable: {exc}"
        return report
    if not bool(getattr(api_config, "API_ENABLED", False)):
        report["error"] = "API_ENABLED is false"
        return report
    image_path = Path(str(payload.get("image", "")))
    image = None
    if image_path.is_file():
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    overrides = payload.setdefault("_designator_overrides", {})
    used_keys = {
        str(item.get("key"))
        for item in debug.get("components", [])
        if isinstance(item, dict) and item.get("key")
    }
    for task in debug.get("designator_recovery_queue", []):
        if not isinstance(task, dict):
            continue
        report["requested"] += 1
        component_index = task.get("component_index")
        detector_type = str(task.get("detector_type") or "")
        # Every non-coordinate component may have a visible designator.  The
        # local detector type remains authoritative and the resolver applies
        # a type-specific prefix gate; API review is not limited to ICs.
        if (
            detector_type in COORDINATE_MATCHED_TYPES
            or detector_type not in TYPE_DESIGNATOR_PREFIXES
        ):
            report["results"].append(
                {"component_index": component_index, "accepted": False, "skipped": "type_not_designator_bearing"}
            )
            continue
        try:
            result = call_qwen_designator(
                component_type=detector_type,
                component_index=int(component_index),
                ocr_candidates=task.get("candidates", []),
                partial_designator=task.get("partial_designator"),
                image=image,
                bbox_top_left=task.get("bbox_top_left"),
            )
            designator, confidence = accept_qwen_designator(result, detector_type)
            if not designator or designator in used_keys:
                report["failed"] += 1
                report["results"].append(
                    {
                        "component_index": component_index,
                        "accepted": False,
                        "confidence": confidence,
                        "api_designator": result.get("designator"),
                        "api_corrected_text": result.get("corrected_text"),
                        "api_uncertain": result.get("uncertain"),
                        "api_reason": result.get("reason"),
                    }
                )
                continue
            used_keys.add(designator)
            overrides[str(component_index)] = {
                "designator": designator,
                "text": designator,
                "score": confidence,
                "source": "qwen_designator_review",
                "crop_top_left": task.get("bbox_top_left"),
                "ocr_rotation": 0,
                "component_index": component_index,
                "api_result": result,
            }
            report["accepted"] += 1
            report["results"].append(
                {
                    "component_index": component_index,
                    "accepted": True,
                    "designator": designator,
                    "confidence": confidence,
                    "api_uncertain": result.get("uncertain"),
                    "api_reason": result.get("reason"),
                }
            )
        except Exception as exc:
            report["failed"] += 1
            report["results"].append(
                {"component_index": component_index, "accepted": False, "error": str(exc)}
            )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="YOLO+OCR intermediate JSON")
    parser.add_argument("--output", type=Path, required=True, help="Component JSON output")
    parser.add_argument(
        "--debug-output",
        type=Path,
        help="Optional sidecar retaining OCR candidate roles and associations",
    )
    parser.add_argument(
        "--type-map",
        type=Path,
        default=ROOT / "configs" / "type_map.json",
    )
    parser.add_argument("--strict-types", action="store_true")
    parser.add_argument(
        "--local-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="automatically recover visible designators with local OCR (default: true)",
    )
    parser.add_argument(
        "--api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow Qwen to review unresolved designator text after local OCR (default: false)",
    )
    parser.add_argument(
        "--local-ocr-all-small-symbols",
        action="store_true",
        help="also run rotated local OCR on all small R/C/SW/D/Q/X symbols",
    )
    parser.add_argument("--local-ocr-device", default="cpu")
    parser.add_argument(
        "--local-ocr-scales",
        default="2,4",
        help="comma-separated crop upscales used by the unresolved-key OCR pass",
    )
    parser.add_argument(
        "--local-ocr-rotations",
        default="0,90,270",
        help="comma-separated crop rotations used by the unresolved-key OCR pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_intermediate(args.input)
    mapping = load_type_mapping(args.type_map) if args.type_map.is_file() else None
    result, debug = build_components(payload, type_mapping=mapping, strict_types=args.strict_types)
    if args.local_ocr:
        scales = [float(value) for value in args.local_ocr_scales.split(",") if value.strip()]
        local_report = add_local_ocr_evidence(
            payload,
            local_ocr_detection_indices(
                payload,
                debug,
                all_small_symbols=args.local_ocr_all_small_symbols,
            ),
            device=args.local_ocr_device,
            scales=scales or [2.0],
            rotations=[int(value) % 360 for value in args.local_ocr_rotations.split(",") if value.strip()] or [0],
        )
        result, debug = build_components(payload, type_mapping=mapping, strict_types=args.strict_types)
        debug["local_ocr"] = local_report
    if args.api:
        api_report = add_api_designator_evidence(payload, debug, enabled=True)
        result, debug = build_components(payload, type_mapping=mapping, strict_types=args.strict_types)
        debug["designator_api"] = api_report
    errors = validate_submission(
        result,
        image_width=payload["image_size"].get("width"),
        image_height=payload["image_size"].get("height"),
    )
    if errors:
        raise ValueError("Component output failed protocol validation:\n- " + "\n- ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.debug_output:
        args.debug_output.parent.mkdir(parents=True, exist_ok=True)
        args.debug_output.write_text(json.dumps(debug, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"components={len(result['components'])}, pins=0, nets=0 -> {args.output}")


if __name__ == "__main__":
    main()
