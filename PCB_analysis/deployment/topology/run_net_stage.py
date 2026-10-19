#!/usr/bin/env python3
"""Serialize the current wire/pin exploration result as contest JSON nets.

This stage intentionally uses only measured wire segments and measured pin
points.  It does not use Qwen, IC-library ordering, public target JSON, or
fixed-case edge counts to create connections.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.schema_validator import validate_submission  # noqa: E402
from topology.net_builder import build_nets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--pins-json", type=Path, required=True)
    parser.add_argument("--wire-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snap-tolerance", type=float, default=5.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise FileNotFoundError(path)
    return image


def main() -> None:
    args = parse_args()
    if not 0 < args.snap_tolerance <= 5.0:
        raise ValueError("--snap-tolerance must be in (0, 5]; the contest edge tolerance is 5 px")
    image = read_image(args.image)
    pin_payload = read_json(args.pins_json)
    wire_payload = read_json(args.wire_json)
    segments = wire_payload.get("wire_geometry", {}).get("segments_top_left", [])
    if not isinstance(segments, list):
        raise ValueError("wire JSON does not contain wire_geometry.segments_top_left")
    components = pin_payload.get("components", {})
    pins = pin_payload.get("pins", {})
    if not isinstance(components, dict) or not isinstance(pins, dict):
        raise ValueError("pin JSON does not contain formal components/pins objects")
    nets = build_nets(
        pins,
        segments,
        image_height=float(image.shape[0]),
        snap_tolerance=args.snap_tolerance,
    )
    output = {"components": components, "pins": pins, "nets": nets}
    errors = validate_submission(output, image_width=float(image.shape[1]), image_height=float(image.shape[0]))
    if errors:
        raise ValueError("net stage failed protocol validation:\n- " + "\n- ".join(errors))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"nets={len(nets)} -> {args.output}")


if __name__ == "__main__":
    main()
