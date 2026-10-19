#!/usr/bin/env python3
"""Run automatic IC name resolution and IC pin localization as one stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.coordinate import bottom_left_bbox_to_top_left  # noqa: E402
from ic_library.index import load_index  # noqa: E402
from topology.ic_name_resolver import resolve_all  # noqa: E402
from topology.pin_locator import (  # noqa: E402
    locate_ic_pins,
    locate_non_ic_pins,
    read_image,
)
from topology.wire_detector import detect_wire_segments  # noqa: E402
from topology.qwen_resolver import resolve_needs_api  # noqa: E402
from topology.ic_local_ocr import create_local_ocr, recognize_ic_crop  # noqa: E402


def add_unresolved_ic_candidates(
    components: dict[str, Any],
    debug_payload: dict[str, Any] | None,
) -> set[str]:
    """Expose key-unresolved IC detections to the local model-name stage.

    ``__ic_detection_*`` is an internal-only identifier.  It can carry OCR,
    library and geometry diagnostics before a contest-valid component key is
    available, but must never be copied into the final submission JSON.
    """
    internal_keys: set[str] = set()
    if not isinstance(debug_payload, dict):
        return internal_keys
    for item in debug_payload.get("components", []):
        if not isinstance(item, dict) or item.get("type") != "box" or not item.get("skipped"):
            continue
        bbox = item.get("bbox_bottom_left")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            index = int(item.get("component_index"))
        except (TypeError, ValueError):
            continue
        key = f"__ic_detection_{index:03d}"
        if key in components:
            continue
        components[key] = {"Name": key, "type": "box", "bbox": bbox, "value": None}
        internal_keys.add(key)
    return internal_keys


def build_formal_components(
    components: dict[str, Any],
    resolutions: dict[str, Any],
    internal_ic_keys: set[str],
) -> dict[str, Any]:
    """Apply verified IC names without promoting internal unresolved IDs.

    The dictionary key is the visible component designator.  ``Name`` is the
    verified IC model when resolution succeeded; otherwise it remains the
    visible designator.  ``__ic_detection_*`` records are diagnostics only
    and are never copied into the contest component object.
    """
    output: dict[str, Any] = {}
    for key, component in components.items():
        if key in internal_ic_keys or not isinstance(component, dict):
            continue
        item = dict(component)
        if item.get("type") == "box":
            resolution = resolutions.get(key, {})
            model_name = resolution.get("model_name")
            item["Name"] = str(model_name) if resolution.get("status") == "matched" and model_name else key
        output[key] = item
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True, help="YOLO+OCR intermediate JSON")
    parser.add_argument("--component-json", type=Path, required=True)
    parser.add_argument(
        "--component-debug",
        type=Path,
        help="Optional component_builder debug sidecar; lets IC OCR run before an unresolved U?/R?/C? key is finalized.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "ic_library" / "BoxICLibrary_200cases.index.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolver-output", type=Path, help="Optional automatic IC map sidecar")
    parser.add_argument("--context-margin", type=float, default=100.0)
    parser.add_argument(
        "--local-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one padded high-resolution PP-OCR pass for ICs unresolved by page OCR.",
    )
    parser.add_argument("--local-ocr-device", default="cpu")
    parser.add_argument("--expected-offset", type=float, default=12.0)
    parser.add_argument(
        "--api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Qwen only for unresolved IC model-text ambiguity; never for pin geometry.",
    )
    parser.add_argument(
        "--with-global-wire-diagnostics",
        action="store_true",
        help="also extract whole-page wire geometry for debugging; local pin tips do not require it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = read_image(args.image)
    intermediate = json.loads(args.intermediate.read_text(encoding="utf-8"))
    component_payload = json.loads(args.component_json.read_text(encoding="utf-8"))
    components = dict(component_payload.get("components", {}))
    debug_payload = (
        json.loads(args.component_debug.read_text(encoding="utf-8"))
        if args.component_debug and args.component_debug.is_file()
        else None
    )
    internal_ic_keys = add_unresolved_ic_candidates(components, debug_payload)
    index = load_index(args.index)
    resolutions = resolve_all(
        intermediate,
        components,
        index,
        context_margin=args.context_margin,
    )
    if args.local_ocr:
        local_ocr_by_component: dict[str, list[dict[str, Any]]] = {}
        unresolved = [
            key
            for key, resolution in resolutions.items()
            if resolution.get("needs_api") and isinstance(components.get(key), dict)
        ]
        if unresolved:
            try:
                local_ocr = create_local_ocr(device=args.local_ocr_device)
                for component_key in unresolved:
                    component = components[component_key]
                    bbox_top_left = bottom_left_bbox_to_top_left(component["bbox"], float(image.shape[0]))
                    local_ocr_by_component[component_key] = recognize_ic_crop(
                        local_ocr,
                        image,
                        bbox_top_left,
                    )
                resolutions = resolve_all(
                    intermediate,
                    components,
                    index,
                    context_margin=args.context_margin,
                    local_ocr_by_component=local_ocr_by_component,
                )
            except Exception as exc:
                # Full-page OCR remains a valid fallback if the optional
                # local pass cannot be initialized in the contest container.
                for component_key in unresolved:
                    resolutions[component_key]["local_ocr_error"] = str(exc)
    if args.api:
        resolutions = resolve_needs_api(
            resolutions,
            components,
            image,
            float(image.shape[0]),
            index,
        )
    if args.resolver_output:
        args.resolver_output.parent.mkdir(parents=True, exist_ok=True)
        args.resolver_output.write_text(
            json.dumps(resolutions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    image_height = float(image.shape[0])
    ocr_items = intermediate.get("ocr", {}).get("texts", [])
    wire_segments: list[dict[str, Any]] | None = None
    if args.with_global_wire_diagnostics:
        component_boxes_top_left = [
            bottom_left_bbox_to_top_left(component["bbox"], image_height)
            for component in components.values()
            if isinstance(component, dict)
            and isinstance(component.get("bbox"), list)
        ]
        wire_segments = detect_wire_segments(
            image,
            component_boxes_top_left=component_boxes_top_left,
            ocr_items=ocr_items,
        )
    all_pins: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for component_key, component in components.items():
        if not isinstance(component, dict):
            continue
        resolution = resolutions.get(component_key, {})
        bbox_top_left = bottom_left_bbox_to_top_left(component["bbox"], image_height)
        if component.get("type") == "box":
            model_name = resolution.get("model_name")
            record = index.get(model_name) if model_name else None
            pin_result = locate_ic_pins(
                image,
                bbox_top_left,
                record,
                ocr_items,
                image_height=image_height,
                expected_offset=args.expected_offset,
                wire_segments=wire_segments,
            )
        else:
            # Ordinary symbols have no IC-library pin table.  Their points
            # come from local terminal geometry and their pin numbers follow
            # the contest's automatic numbering rule.
            pin_result = locate_non_ic_pins(
                image,
                bbox_top_left,
                str(component.get("type", "")),
                ocr_items,
                image_height=image_height,
            )
        diagnostics[component_key] = {
            "resolution": resolution,
            "pin_locator": pin_result["diagnostics"],
        }
        if pin_result["pins"]:
            if component_key in internal_ic_keys:
                # Geometry/OCR diagnostics for an unresolved component are
                # useful for later review, but an internal key is not a valid
                # contest component key and must not enter formal ``pins``.
                diagnostics.setdefault(component_key, {})["internal_pins"] = pin_result["pins"]
            else:
                all_pins[component_key] = pin_result["pins"]
    formal_components = build_formal_components(components, resolutions, internal_ic_keys)
    output = {
        "coordinate_system": "bottom_left",
        "components": formal_components,
        "pins": all_pins,
        "wire_geometry_top_left": wire_segments or [],
        "ic_resolution": resolutions,
        "diagnostics": diagnostics,
        "internal_ic_candidate_keys": sorted(internal_ic_keys),
        "note": (
            "This stage outputs pin data only. Internal __ic_detection_* keys are diagnostics "
            "for unresolved designators and must be rekeyed from explicit evidence before final JSON validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    matched = sum(1 for item in resolutions.values() if item.get("status") == "matched")
    print(f"ICs={len(resolutions)}, matched={matched}, pin_components={len(all_pins)} -> {args.output}")


if __name__ == "__main__":
    main()
