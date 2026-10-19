"""YOLO-to-official Problem 6 component type normalization."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path


# Exact vocabulary from the updated official ``type_list.md``. Detector-only
# aliases are normalized by DEFAULT_TYPE_MAP below.
DEFAULT_ALLOWED_TYPES = frozenset(
    {
        "amp",
        "ant",
        "battary",
        "bjt",
        "block",
        "box",
        "buzzer",
        "c",
        "circle_header",
        "crystal",
        "d",
        "dc",
        "esd",
        "fuse",
        "gnd",
        "jumper",
        "l",
        "led",
        "m3螺丝",
        "motor",
        "mosfet",
        "net_bidirection",
        "net_input",
        "net_output",
        "net_short",
        "opt",
        "other",
        "pin",
        "r",
        "seg",
        "switch",
        "testpoint",
        "transfomer",
        "v",
    }
)


DEFAULT_TYPE_MAP = {
    # Older documents and internal code use these IC aliases.
    "ic": "box",
    "boxic": "box",
    # YOLO fine-grained labels collapse to the official generic types.
    "amp_3pin": "amp",
    "amp_5pin": "amp",
    "bjt_npn": "bjt",
    "bjt_pnp": "bjt",
    "crystal_2pin": "crystal",
    "crystal_3pin": "crystal",
    "crystal_4pin": "crystal",
    "mosfet_npn": "mosfet",
    "mosfet_pnp": "mosfet",
    "m3_screw": "m3螺丝",
}


def load_type_mapping(path: str | Path) -> dict[str, str]:
    """Load an optional JSON mapping maintained outside detector code."""
    mapping_path = Path(path)
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"type mapping must be a JSON object: {mapping_path}")
    return {str(key).lower(): str(value).lower() for key, value in payload.items()}


def map_component_type(
    raw_type: str,
    mapping: Mapping[str, str] | None = None,
    *,
    allowed_types: set[str] | frozenset[str] | None = None,
    strict: bool = False,
) -> str:
    """Normalize one detector type and optionally enforce the official list."""
    if not isinstance(raw_type, str) or not raw_type.strip():
        raise ValueError("component type must be a non-empty string")
    normalized = raw_type.strip().lower()
    aliases = dict(DEFAULT_TYPE_MAP)
    if mapping:
        aliases.update({str(key).lower(): str(value) for key, value in mapping.items()})
    result = aliases.get(normalized, normalized)
    allowed = DEFAULT_ALLOWED_TYPES if allowed_types is None else allowed_types
    if strict and result not in allowed:
        raise ValueError(f"unsupported component type: {raw_type!r} -> {result!r}")
    return result
