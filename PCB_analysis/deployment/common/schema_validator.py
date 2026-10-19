"""Structural validator for the Problem 6 final JSON protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from .type_mapper import DEFAULT_ALLOWED_TYPES


PIN_KEY_RE = re.compile(r"^pin_.+$")


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_point(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and _is_number(value.get("x"))
        and _is_number(value.get("y"))
    )


def _is_bbox(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 4
        and all(_is_number(item) for item in value)
        and value[0] <= value[2]
        and value[1] <= value[3]
    )


def _add(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def validate_submission(
    payload: Any,
    *,
    allowed_types: set[str] | frozenset[str] | None = None,
    image_width: float | None = None,
    image_height: float | None = None,
) -> list[str]:
    """Return structural validation errors; an empty list means valid."""
    errors: list[str] = []
    allowed = DEFAULT_ALLOWED_TYPES if allowed_types is None else allowed_types

    if not isinstance(payload, Mapping):
        return ["$: final payload must be an object"]
    for required in ("components", "pins", "nets"):
        if required not in payload:
            _add(errors, "$", f"missing required field {required!r}")
    if errors:
        return errors

    components = payload["components"]
    pins = payload["pins"]
    nets = payload["nets"]
    if not isinstance(components, Mapping):
        _add(errors, "$.components", "must be an object")
        components = {}
    if not isinstance(pins, Mapping):
        _add(errors, "$.pins", "must be an object")
        pins = {}
    if not isinstance(nets, Mapping):
        _add(errors, "$.nets", "must be an object")
        nets = {}

    for component_key, component in components.items():
        path = f"$.components[{component_key!r}]"
        if not isinstance(component_key, str) or not component_key:
            _add(errors, path, "component key must be a non-empty string")
        if not isinstance(component, Mapping):
            _add(errors, path, "must be an object")
            continue
        if not isinstance(component.get("Name"), str) or not component["Name"].strip():
            _add(errors, f"{path}.Name", "must be a non-empty string")
        component_type = component.get("type")
        if not isinstance(component_type, str) or not component_type.strip():
            _add(errors, f"{path}.type", "must be a non-empty string")
        elif component_type not in allowed:
            _add(errors, f"{path}.type", f"unsupported type {component_type!r}")
        value = component.get("value")
        if value is not None and not isinstance(value, (str, int, float)):
            _add(errors, f"{path}.value", "must be string, number or null")
        bbox = component.get("bbox")
        if not _is_bbox(bbox):
            _add(errors, f"{path}.bbox", "must be [x1,y1,x2,y2] with ordered numbers")
        if image_width is not None and (bbox[0] < 0 or bbox[2] > image_width):
            _add(errors, f"{path}.bbox", "x coordinates exceed image bounds")
        if image_height is not None and (bbox[1] < 0 or bbox[3] > image_height):
            _add(errors, f"{path}.bbox", "y coordinates exceed image bounds")

    component_keys = set(components.keys())
    for component_key, component_pins in pins.items():
        path = f"$.pins[{component_key!r}]"
        if component_key not in component_keys:
            _add(errors, path, "has no matching component")
        if not isinstance(component_pins, Mapping):
            _add(errors, path, "must be an object")
            continue
        for pin_key, pin in component_pins.items():
            pin_path = f"{path}[{pin_key!r}]"
            if not isinstance(pin_key, str) or not PIN_KEY_RE.match(pin_key):
                _add(errors, pin_path, "pin key must use the pin_<number> format")
            if not isinstance(pin, Mapping):
                _add(errors, pin_path, "must be an object")
                continue
            if pin.get("pinname") is not None and not isinstance(pin.get("pinname"), str):
                _add(errors, f"{pin_path}.pinname", "must be a string or null")
            if not _is_point(pin.get("point")):
                _add(errors, f"{pin_path}.point", "must be an object with numeric x and y")
            if image_width is not None and not 0 <= pin["point"]["x"] <= image_width:
                _add(errors, f"{pin_path}.point.x", "is outside image bounds")
            if image_height is not None and not 0 <= pin["point"]["y"] <= image_height:
                _add(errors, f"{pin_path}.point.y", "is outside image bounds")

    for net_key, net in nets.items():
        path = f"$.nets[{net_key!r}]"
        if not isinstance(net, Mapping):
            _add(errors, path, "must be an object")
            continue
        if not isinstance(net.get("hyperGraph"), str):
            _add(errors, f"{path}.hyperGraph", "must be a string")
        else:
            _validate_hypergraph_refs(net["hyperGraph"], path, pins, errors)
        edges = net.get("edges")
        if not isinstance(edges, Mapping):
            _add(errors, f"{path}.edges", "must be an object")
            continue
        for edge_key, edge in edges.items():
            edge_path = f"{path}.edges[{edge_key!r}]"
            if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)) or len(edge) != 2:
                _add(errors, edge_path, "must contain exactly two points")
                continue
            for index, point in enumerate(edge):
                if not _is_point(point):
                    _add(errors, f"{edge_path}[{index}]", "must be an object with numeric x and y")

    return errors


def _validate_hypergraph_refs(
    hypergraph: str,
    path: str,
    pins: Mapping[str, Any],
    errors: list[str],
) -> None:
    text = hypergraph.strip()
    if not (text.startswith("(") and text.endswith(")")):
        _add(errors, f"{path}.hyperGraph", "must be enclosed in parentheses")
        return
    refs = [item.strip() for item in text[1:-1].split(",") if item.strip()]
    for ref in refs:
        if "." not in ref:
            _add(errors, f"{path}.hyperGraph", f"invalid pin reference {ref!r}")
            continue
        component_key, pin_number = ref.rsplit(".", 1)
        # Power/flywire labels such as ∅100.97 are not component pin refs.
        if component_key not in pins:
            continue
        expected_key = f"pin_{pin_number}"
        if expected_key not in pins[component_key]:
            _add(errors, f"{path}.hyperGraph", f"unknown pin reference {ref!r}")


def assert_valid_submission(payload: Any, **kwargs: Any) -> None:
    errors = validate_submission(payload, **kwargs)
    if errors:
        raise ValueError("Invalid Problem 6 JSON:\n- " + "\n- ".join(errors))
