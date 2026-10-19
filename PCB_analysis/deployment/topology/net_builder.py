"""Build contest nets from measured wire segments and pin points."""

from __future__ import annotations

import math
from typing import Any

from common.coordinate import top_left_point_to_bottom_left


def _point(value: dict[str, Any]) -> tuple[float, float]:
    return float(value["x"]), float(value["y"])


def _distance_to_segment(point: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class _DSU:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def add(self) -> int:
        index = len(self.parent)
        self.parent.append(index)
        return index

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def build_nets(
    pins: dict[str, dict[str, Any]],
    wire_segments: list[dict[str, Any]],
    *,
    image_height: float,
    snap_tolerance: float = 5.0,
) -> dict[str, dict[str, Any]]:
    """Connect pins to measured edges using a tolerance no larger than 5 px."""
    dsu = _DSU()
    node_points: list[tuple[float, float]] = []
    node_edges: list[tuple[int, int, dict[str, Any]]] = []
    pin_nodes: dict[tuple[str, str], int] = {}

    def node(point: tuple[float, float]) -> int:
        for index, existing in enumerate(node_points):
            if math.hypot(point[0] - existing[0], point[1] - existing[1]) <= snap_tolerance:
                return index
        index = dsu.add()
        node_points.append(point)
        return index

    for segment in wire_segments:
        values = segment.get("points_top_left", [])
        if len(values) != 2:
            continue
        a, b = _point(values[0]), _point(values[1])
        left, right = node(a), node(b)
        dsu.union(left, right)
        node_edges.append((left, right, segment))

    for component_key, component_pins in pins.items():
        for pin_key, pin in component_pins.items():
            point_bl = pin.get("point")
            if not isinstance(point_bl, dict):
                continue
            point = (float(point_bl["x"]), float(image_height) - float(point_bl["y"]))
            pin_index = node(point)
            pin_nodes[(component_key, pin_key)] = pin_index
            for left, right, segment in node_edges:
                a, b = _point(segment["points_top_left"][0]), _point(segment["points_top_left"][1])
                if _distance_to_segment(point, a, b) <= snap_tolerance:
                    dsu.union(pin_index, left)
                    dsu.union(pin_index, right)

    groups: dict[int, dict[str, Any]] = {}
    for (component_key, pin_key), index in pin_nodes.items():
        root = dsu.find(index)
        group = groups.setdefault(root, {"pins": [], "edges": []})
        number = pin_key.removeprefix("pin_")
        group["pins"].append(f"{component_key}.{number}")

    for left, right, segment in node_edges:
        root = dsu.find(left)
        if root not in groups:
            continue
        group = groups[root]
        p1, p2 = segment["points_top_left"]
        group["edges"].append((p1, p2))

    nets: dict[str, dict[str, Any]] = {}
    for number, group in enumerate(groups.values(), start=1):
        if not group["pins"]:
            continue
        edges: dict[str, list[dict[str, float]]] = {}
        for edge_index, (p1, p2) in enumerate(group["edges"], start=1):
            edges[f"edge_{edge_index}"] = [
                top_left_point_to_bottom_left(p1, image_height, decimals=2),
                top_left_point_to_bottom_left(p2, image_height, decimals=2),
            ]
        nets[f"net_{number}"] = {
            "hyperGraph": "(" + ",".join(sorted(set(group["pins"]))) + ")",
            "edges": edges,
        }
    return nets
