#!/usr/bin/env python3
"""Run the contest wire-geometry stage.

This stage intentionally stops before net assignment.  It produces:

* skeleton-derived two-point wire segments in native top-left coordinates;
* component-terminal and junction candidates;
* protocol-compatible bottom-left ``edge_i`` entries for later net_builder use.

No component key, IC library order or Qwen result is used to invent wire
coordinates.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.coordinate import (  # noqa: E402
    bottom_left_bbox_to_top_left,
    bottom_left_point_to_top_left,
    top_left_point_to_bottom_left,
)
from topology.pin_locator import read_image  # noqa: E402
from topology.wire_detector import (  # noqa: E402
    detect_wire_geometry,
    segments_to_submission_edges,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def component_boxes_top_left(payload: dict[str, Any], image_height: float) -> list[list[float]]:
    components = payload.get("components", payload)
    if not isinstance(components, dict):
        return []
    boxes: list[list[float]] = []
    for item in components.values():
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            boxes.append(bottom_left_bbox_to_top_left(bbox, image_height, decimals=2))
    return boxes


def merge_unresolved_ic_geometry(
    components_payload: dict[str, Any],
    pin_payload: dict[str, Any],
    image_height: float,
) -> dict[str, Any]:
    """Add unresolved IC boxes that already have local geometry evidence.

    The semantic pin stage may keep an unresolved IC out of the formal
    component map, while its diagnostics still contain the YOLO/local-ROI
    bbox.  Wire extraction must retain that measured geometry.  No pin order,
    library record, or guessed coordinate is created here.
    """
    output = deepcopy(components_payload)
    components = output.get("components", output)
    if not isinstance(components, dict):
        return output
    diagnostics = pin_payload.get("diagnostics", {})
    for key in pin_payload.get("internal_ic_candidate_keys", []):
        if not isinstance(key, str) or key in components:
            continue
        record = diagnostics.get(key, {}) if isinstance(diagnostics, dict) else {}
        locator = record.get("pin_locator", {}) if isinstance(record, dict) else {}
        bbox = locator.get("bbox_used_top_left") or locator.get("bbox_top_left")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            continue
        components[key] = {
            "Name": key,
            "type": "box",
            "value": None,
            "bbox": [x1, image_height - y2, x2, image_height - y1],
            "geometry_source": "pin_stage_local_diagnostic_bbox",
        }
    return output


def bottom_left_terminal_records(records: list[dict[str, Any]], image_height: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        point = item.get("point_top_left")
        skeleton = item.get("skeleton_point_top_left")
        reference = item.get("component_boundary_reference_top_left")
        if isinstance(point, dict):
            item["point"] = top_left_point_to_bottom_left(point, image_height, decimals=2)
        if isinstance(skeleton, dict):
            item["skeleton_point"] = top_left_point_to_bottom_left(skeleton, image_height, decimals=2)
        if isinstance(reference, dict):
            item["component_boundary_reference"] = top_left_point_to_bottom_left(
                reference, image_height, decimals=2
            )
        output.append(item)
    return output


def bottom_left_junction_records(records: list[dict[str, Any]], image_height: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        point = item.get("point_top_left")
        if isinstance(point, dict):
            item["point"] = top_left_point_to_bottom_left(point, image_height, decimals=2)
        output.append(item)
    return output


def pin_anchors_top_left(
    payload: dict[str, Any],
    components_payload: dict[str, Any],
    image_height: float,
) -> list[dict[str, Any]]:
    """Convert formal pin points into wire anchors without inventing edges."""
    pins = payload.get("pins", {})
    components = components_payload.get("components", components_payload)
    if not isinstance(pins, dict) or not isinstance(components, dict):
        return []
    pins_for_geometry = dict(pins)
    diagnostics = payload.get("diagnostics", {})
    # Unresolved ICs can still have image-supported local pin candidates.
    # Promote those candidates only as geometric wire anchors; identity and
    # final pin fields remain the responsibility of the pin stage.
    for key in payload.get("internal_ic_candidate_keys", []):
        if not isinstance(key, str) or key in pins_for_geometry:
            continue
        record = diagnostics.get(key, {}) if isinstance(diagnostics, dict) else {}
        locator = record.get("pin_locator", {}) if isinstance(record, dict) else {}
        candidates = locator.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        recovered: dict[str, Any] = {}
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            point = candidate.get("point_top_left")
            if not isinstance(point, dict):
                continue
            try:
                x, y = float(point["x"]), float(point["y"])
            except (KeyError, TypeError, ValueError):
                continue
            number = str(candidate.get("number") or index)
            pin_key = f"pin_{number}"
            while pin_key in recovered:
                index += 1
                pin_key = f"pin_{index}"
            recovered[pin_key] = {
                "pinname": candidate.get("visible_name_evidence")
                or candidate.get("name_evidence")
                or "",
                "point": {"x": x, "y": float(image_height) - y},
                "geometry_source": "pin_stage_local_diagnostic_candidate",
            }
        if recovered:
            pins_for_geometry[key] = recovered
    component_keys = list(components)
    output: list[dict[str, Any]] = []
    for component_key, component_pins in pins_for_geometry.items():
        if not isinstance(component_pins, dict):
            continue
        try:
            component_index = component_keys.index(component_key)
        except ValueError:
            component_index = None
        component_box = None
        if component_index is not None:
            raw_component = components.get(component_key)
            if isinstance(raw_component, dict) and isinstance(raw_component.get("bbox"), list):
                component_box = bottom_left_bbox_to_top_left(
                    raw_component["bbox"], image_height, decimals=2
                )
        for pin_key, pin in component_pins.items():
            if not isinstance(pin, dict) or not isinstance(pin.get("point"), dict):
                continue
            try:
                point = bottom_left_point_to_top_left(pin["point"], image_height, decimals=2)
            except (TypeError, ValueError, KeyError):
                continue
            output.append(
                {
                    "component_key": component_key,
                    "pin_key": pin_key,
                    "component_index": component_index,
                    "side": (
                        "left"
                        if component_box and point["x"] <= component_box[0]
                        else "right"
                        if component_box and point["x"] >= component_box[2]
                        else "top"
                        if component_box and point["y"] <= component_box[1]
                        else "bottom"
                        if component_box and point["y"] >= component_box[3]
                        else "unknown"
                    ),
                    "point_top_left": point,
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--components-json", type=Path, required=True)
    parser.add_argument(
        "--pins-json",
        type=Path,
        help="Optional pin-stage JSON; points are anchors only when image skeleton evidence exists.",
    )
    parser.add_argument(
        "--intermediate",
        type=Path,
        help="Optional YOLO+OCR JSON; OCR is used only for posterior evidence.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image = read_image(args.image)
    components = read_json(args.components_json)
    intermediate = read_json(args.intermediate) if args.intermediate else {}
    pin_payload = read_json(args.pins_json) if args.pins_json else {}
    ocr_items = intermediate.get("ocr", {}).get("texts", [])
    height = float(image.shape[0])
    wire_components = merge_unresolved_ic_geometry(components, pin_payload, height)
    boxes = component_boxes_top_left(wire_components, height)
    anchors = pin_anchors_top_left(pin_payload, wire_components, height)
    geometry = detect_wire_geometry(
        image,
        component_boxes_top_left=boxes,
        ocr_items=ocr_items,
        pin_anchors_top_left=anchors,
    )
    segments = geometry["segments_top_left"]
    output = {
        "coordinate_system": "bottom_left",
        "wire_geometry": {
            "segments_top_left": segments,
            "edges": segments_to_submission_edges(segments, height),
            "component_terminals": bottom_left_terminal_records(
                geometry["component_terminals_top_left"], height
            ),
            "junction_candidates": bottom_left_junction_records(
                geometry["junction_candidates_top_left"], height
            ),
        },
        "diagnostics": {
            key: value
            for key, value in geometry.items()
            if key not in {"segments_top_left", "component_terminals_top_left", "junction_candidates_top_left"}
        },
        "note": (
            "This is the wire geometry stage. The edges must be assigned to "
            "nets[*].edges only after pin/component connectivity is built."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"segments={len(segments)}, terminals={len(geometry['component_terminals_top_left'])}, "
        f"junction_candidates={len(geometry['junction_candidates_top_left'])} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
