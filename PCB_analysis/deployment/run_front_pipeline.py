#!/usr/bin/env python3
"""Unified entry point for the currently runnable Problem 6 pipeline.

Stages are deliberately switchable so that the current exploratory wire/net
implementation is preserved without being mistaken for a stable dependency of
component detection:

    ONNX YOLO -> PP-OCRv6 -> protocol/component assembly
                          -> optional IC name + pin localization
                          -> optional wire geometry -> optional net assembly

All local vision coordinates are top-left during inference.  Component, pin
and net JSON is serialized in the contest bottom-left coordinate system.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.schema_validator import validate_submission  # noqa: E402
from common.type_mapper import load_type_mapping  # noqa: E402
from run_yolo_ocr import (  # noqa: E402
    associate_texts,
    create_ocr,
    create_yolo_session,
    get_image_paths,
    read_image,
    read_ocr_items,
    run_one,
)
from stages.component_builder import (  # noqa: E402
    add_api_designator_evidence,
    add_local_ocr_evidence,
    build_components,
    load_intermediate,
    local_ocr_detection_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", type=Path, action="append", help="One or more schematic images")
    inputs.add_argument("--input-dir", type=Path, help="Directory containing schematic images")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "front_pipeline")

    parser.add_argument("--yolo-model", type=Path, default=ROOT / "onnx_yolo" / "models" / "best.onnx")
    parser.add_argument("--classes", type=Path, default=ROOT / "onnx_yolo" / "classes.json")
    parser.add_argument("--yolo-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--enable-tiles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-size", type=int, default=960)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--tile-fusion-iou", type=float, default=0.55)
    parser.add_argument("--tile-suppression-iou", type=float, default=0.30)
    parser.add_argument("--tile-supplement-conf", type=float, default=0.30)
    parser.add_argument("--tile-single-view-conf", type=float, default=0.45)
    parser.add_argument("--tile-consistency-iou", type=float, default=0.50)
    parser.add_argument("--tile-boundary-margin-ratio", type=float, default=0.01)
    parser.add_argument("--adaptive-tile-size", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-min-size", type=int, default=384)
    parser.add_argument("--small-object-classes", default="r,c,gnd,v")
    parser.add_argument("--small-object-conf", type=float, default=0.12)

    parser.add_argument("--ocr-device", default="cpu")
    parser.add_argument("--det-model-dir", type=Path, default=ROOT / "ppocrv6" / "models" / "det")
    parser.add_argument("--rec-model-dir", type=Path, default=ROOT / "ppocrv6" / "models" / "rec")
    parser.add_argument("--ocr-box-thresh", type=float, default=0.30)
    parser.add_argument("--ocr-rec-score-thresh", type=float, default=0.0)
    parser.add_argument("--ocr-limit-side-len", type=int, default=4000)
    parser.add_argument("--textline-orientation", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--type-map", type=Path, default=ROOT / "configs" / "type_map.json")
    parser.add_argument(
        "--component-local-ocr",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional designator recovery after full-page OCR; disabled by default for predictable CPU runtime.",
    )
    parser.add_argument("--component-local-ocr-all-small-symbols", action="store_true")
    parser.add_argument("--component-local-ocr-device", default="cpu")
    parser.add_argument("--component-local-ocr-scales", default="2,4")
    parser.add_argument("--component-local-ocr-rotations", default="0,90,270")
    parser.add_argument(
        "--component-api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Qwen only for unresolved visible designator text; requires ignored api_config.py.",
    )

    parser.add_argument("--with-ic-pins", action="store_true")
    parser.add_argument("--ic-local-ocr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ic-api", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--with-wire", action="store_true")
    parser.add_argument("--with-net", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def component_stage(args: argparse.Namespace, intermediate_path: Path, output_path: Path, debug_path: Path) -> None:
    payload = load_intermediate(intermediate_path)
    mapping = load_type_mapping(args.type_map) if args.type_map.is_file() else None
    result, debug = build_components(payload, type_mapping=mapping, strict_types=True)

    if args.component_local_ocr:
        scales = [float(value) for value in args.component_local_ocr_scales.split(",") if value.strip()]
        rotations = [int(value) % 360 for value in args.component_local_ocr_rotations.split(",") if value.strip()]
        local_ocr_report = add_local_ocr_evidence(
            payload,
            local_ocr_detection_indices(
                payload,
                debug,
                all_small_symbols=args.component_local_ocr_all_small_symbols,
            ),
            device=args.component_local_ocr_device,
            scales=scales or [2.0],
            rotations=rotations or [0],
        )
        result, debug = build_components(payload, type_mapping=mapping, strict_types=True)
        debug["local_ocr"] = local_ocr_report

    if args.component_api:
        api_report = add_api_designator_evidence(payload, debug, enabled=True)
        result, debug = build_components(payload, type_mapping=mapping, strict_types=True)
        debug["designator_api"] = api_report

    errors = validate_submission(
        result,
        image_width=payload["image_size"].get("width"),
        image_height=payload["image_size"].get("height"),
    )
    if errors:
        raise ValueError("component stage failed protocol validation:\n- " + "\n- ".join(errors))
    write_json(output_path, result)
    write_json(debug_path, debug)


def call_stage(command: list[str]) -> None:
    printable = " ".join(f'"{item}"' if " " in item else item for item in command)
    print(f"[stage] {printable}")
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    if args.with_net and not (args.with_ic_pins and args.with_wire):
        raise ValueError("--with-net requires both --with-ic-pins and --with-wire")
    paths = get_image_paths(args)
    if not args.yolo_model.is_file() or not args.classes.is_file():
        raise FileNotFoundError(f"YOLO model/classes missing: {args.yolo_model}, {args.classes}")
    if not args.det_model_dir.is_dir() or not args.rec_model_dir.is_dir():
        raise FileNotFoundError("PP-OCRv6 detection/recognition model directories are missing")

    names = json.loads(args.classes.read_text(encoding="utf-8"))
    yolo_session, yolo_input_name = create_yolo_session(args.yolo_model, args.yolo_device)
    ocr = create_ocr(
        det_model_dir=args.det_model_dir,
        rec_model_dir=args.rec_model_dir,
        device=args.ocr_device,
        box_thresh=args.ocr_box_thresh,
        rec_score_thresh=args.ocr_rec_score_thresh,
        limit_side_len=args.ocr_limit_side_len,
        textline_orientation=args.textline_orientation,
    )

    run_manifest: dict[str, Any] = {
        "pipeline": "eda6_front_pipeline_v1",
        "stages": ["onnx_yolo", "ppocrv6", "protocol_component_assembly"],
        "optional_stages": {"ic_pins": args.with_ic_pins, "wire": args.with_wire, "net": args.with_net},
        "coordinate_convention": "internal top-left; serialized contest JSON bottom-left",
        "images": [],
    }
    if args.with_ic_pins:
        run_manifest["stages"].append("ic_name_and_pin_localization")
    if args.with_wire:
        run_manifest["stages"].append("wire_geometry_exploratory")
    if args.with_net:
        run_manifest["stages"].append("net_assembly_exploratory")

    for image_path in paths:
        stem = image_path.stem
        intermediate_dir = args.output_dir / "intermediate"
        component_dir = args.output_dir / "components"
        pin_dir = args.output_dir / "ic_pins"
        wire_dir = args.output_dir / "wire"
        final_dir = args.output_dir / "final"
        intermediate_path = run_one(image_path, intermediate_dir, yolo_session, yolo_input_name, ocr, names, args)
        component_path = component_dir / f"{stem}_components.json"
        component_debug_path = component_dir / f"{stem}_component_debug.json"
        component_stage(args, intermediate_path, component_path, component_debug_path)

        result_paths: dict[str, str] = {
            "intermediate": str(intermediate_path),
            "components": str(component_path),
            "component_debug": str(component_debug_path),
        }
        pin_path: Path | None = None
        wire_path: Path | None = None
        if args.with_ic_pins:
            pin_path = pin_dir / f"{stem}_ic_pins.json"
            resolver_path = pin_dir / f"{stem}_ic_resolution.json"
            command = [
                sys.executable,
                str(ROOT / "topology" / "run_ic_pin_stage.py"),
                "--image", str(image_path),
                "--intermediate", str(intermediate_path),
                "--component-json", str(component_path),
                "--component-debug", str(component_debug_path),
                "--resolver-output", str(resolver_path),
                "--output", str(pin_path),
            ]
            command.append("--local-ocr" if args.ic_local_ocr else "--no-local-ocr")
            command.append("--api" if args.ic_api else "--no-api")
            call_stage(command)
            result_paths.update({"ic_pins": str(pin_path), "ic_resolution": str(resolver_path)})

        if args.with_wire:
            wire_path = wire_dir / f"{stem}_wire.json"
            command = [
                sys.executable,
                str(ROOT / "topology" / "run_wire_stage.py"),
                "--image", str(image_path),
                "--components-json", str(component_path),
                "--intermediate", str(intermediate_path),
                "--output", str(wire_path),
            ]
            if pin_path:
                command.extend(["--pins-json", str(pin_path)])
            call_stage(command)
            result_paths["wire"] = str(wire_path)

        if args.with_net and pin_path and wire_path:
            final_path = final_dir / f"{stem}_submission.json"
            call_stage(
                [
                    sys.executable,
                    str(ROOT / "topology" / "run_net_stage.py"),
                    "--image", str(image_path),
                    "--pins-json", str(pin_path),
                    "--wire-json", str(wire_path),
                    "--output", str(final_path),
                ]
            )
            result_paths["submission"] = str(final_path)
        run_manifest["images"].append({"image": str(image_path), "outputs": result_paths})
        print(f"[done] {image_path.name} -> {component_path}")

    write_json(args.output_dir / "run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()
