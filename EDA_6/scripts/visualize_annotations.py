#!/usr/bin/env python3
"""
Visualize PCB schematic annotations from the public training dataset.

The script reads each case folder containing a PNG schematic and its target JSON,
then draws component bounding boxes, pin points, and net edge segments back onto
the original image.

Important:
    The contest JSON uses the original ParsedJson coordinate system:
    origin at the bottom-left, x increasing rightward, y increasing upward.
    This script converts those coordinates to normal image coordinates before
    drawing: origin at the top-left, x increasing rightward, y increasing
    downward. Keep --coord-system bottom-left for the provided dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


TYPE_COLORS = {
    "box": "#1f77b4",
    "ic": "#1f77b4",
    "r": "#d62728",
    "c": "#2ca02c",
    "l": "#9467bd",
    "gnd": "#111111",
    "v": "#ff7f0e",
    "net_input": "#17becf",
    "net_output": "#bcbd22",
    "net_short": "#8c564b",
    "pin": "#e377c2",
    "d": "#7f7f7f",
    "led": "#e41a1c",
    "switch": "#4daf4a",
    "mosfet": "#984ea3",
    "bjt": "#377eb8",
    "other": "#555555",
}


def find_dataset_root(start: Path) -> Path:
    """Find the directory that contains the dataset subsets."""
    candidates = [start, *start.iterdir()]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "200_train_cases").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find dataset root containing '200_train_cases'. "
        "Pass it explicitly with --dataset-root."
    )


def load_font(size: int) -> ImageFont.ImageFont:
    """Load a readable font, falling back to PIL default."""
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for font_path in font_candidates:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def hex_to_rgba(hex_color: str, alpha: int) -> Tuple[int, int, int, int]:
    value = hex_color.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
        alpha,
    )


def color_for_type(component_type: Optional[str]) -> str:
    if not component_type:
        return TYPE_COLORS["other"]
    component_type = str(component_type)
    return TYPE_COLORS.get(component_type, TYPE_COLORS["other"])


def color_for_net(net_name: str) -> str:
    digest = hashlib.md5(net_name.encode("utf-8")).hexdigest()
    r = 80 + int(digest[0:2], 16) % 150
    g = 80 + int(digest[2:4], 16) % 150
    b = 80 + int(digest[4:6], 16) % 150
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    bg: Tuple[int, int, int, int] = (255, 255, 255, 210),
) -> None:
    if not text:
        return
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 2
    bg_box = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    draw.rounded_rectangle(bg_box, radius=2, fill=bg)
    draw.text((x, y), text, font=font, fill=fill)


def iter_cases(subset_dir: Path, selected_case: Optional[str]) -> Iterable[Path]:
    if selected_case:
        case_dir = subset_dir / selected_case
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")
        yield case_dir
        return
    yield from sorted(p for p in subset_dir.iterdir() if p.is_dir())


def find_pair(case_dir: Path) -> Tuple[Path, Path]:
    images = sorted(
        p
        for p in case_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    jsons = sorted(p for p in case_dir.iterdir() if p.suffix.lower() == ".json")
    if not images:
        raise FileNotFoundError(f"No image found in {case_dir}")
    if not jsons:
        raise FileNotFoundError(f"No JSON annotation found in {case_dir}")
    return images[0], jsons[0]


def as_bbox(value: Any) -> Optional[BBox]:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
        return x1, y1, x2, y2
    except (TypeError, ValueError):
        return None


def as_point(value: Any) -> Optional[Point]:
    if not isinstance(value, dict):
        return None
    try:
        return float(value["x"]), float(value["y"])
    except (KeyError, TypeError, ValueError):
        return None


def transform_point(point: Point, image_height: int, coord_system: str) -> Point:
    """Convert annotation point coordinates to PIL image coordinates."""
    x, y = point
    if coord_system == "bottom-left":
        return x, image_height - y
    return x, y


def transform_bbox(bbox: BBox, image_height: int, coord_system: str) -> BBox:
    """Convert annotation bbox coordinates to PIL image coordinates."""
    x1, y1, x2, y2 = bbox
    if coord_system == "bottom-left":
        return x1, image_height - y2, x2, image_height - y1
    return x1, y1, x2, y2


def draw_components(
    overlay: Image.Image,
    data: Dict[str, Any],
    font: ImageFont.ImageFont,
    show_values: bool,
    coord_system: str,
) -> int:
    draw = ImageDraw.Draw(overlay, "RGBA")
    image_height = overlay.height
    count = 0
    for key, component in (data.get("components") or {}).items():
        bbox = as_bbox(component.get("bbox"))
        if not bbox:
            continue
        x1, y1, x2, y2 = transform_bbox(bbox, image_height, coord_system)
        comp_type = component.get("type")
        color = color_for_type(comp_type)
        rgba = hex_to_rgba(color, 235)
        draw.rectangle((x1, y1, x2, y2), outline=rgba, width=2)

        label = f"{key}:{comp_type}"
        if show_values:
            name = component.get("Name")
            value = component.get("value")
            extra = value if value not in (None, "") else name
            if extra not in (None, "", key):
                label += f" {extra}"
        draw_label(draw, (int(x1), max(0, int(y1) - 14)), label, font, color)
        count += 1
    return count


def draw_pins(
    overlay: Image.Image,
    data: Dict[str, Any],
    font: ImageFont.ImageFont,
    show_pin_labels: bool,
    coord_system: str,
) -> int:
    draw = ImageDraw.Draw(overlay, "RGBA")
    image_height = overlay.height
    count = 0
    radius = 3
    for component_key, pins in (data.get("pins") or {}).items():
        if not isinstance(pins, dict):
            continue
        for pin_key, pin in pins.items():
            point = as_point((pin or {}).get("point"))
            if not point:
                continue
            x, y = transform_point(point, image_height, coord_system)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 215, 0, 240),
                outline=(0, 0, 0, 230),
                width=1,
            )
            if show_pin_labels:
                pin_number = pin_key.removeprefix("pin_")
                pinname = pin.get("pinname") if isinstance(pin, dict) else ""
                label = f"{component_key}.{pin_number}"
                if pinname:
                    label += f" {pinname}"
                draw_label(draw, (int(x) + 5, int(y) - 6), label, font, "#111111")
            count += 1
    return count


def draw_nets(
    overlay: Image.Image,
    data: Dict[str, Any],
    font: ImageFont.ImageFont,
    show_net_labels: bool,
    coord_system: str,
) -> int:
    draw = ImageDraw.Draw(overlay, "RGBA")
    image_height = overlay.height
    count = 0
    for net_name, net in (data.get("nets") or {}).items():
        edges = (net or {}).get("edges") or {}
        if not isinstance(edges, dict):
            continue
        color = color_for_net(str(net_name))
        rgba = hex_to_rgba(color, 210)
        first_label_done = False
        for edge_key, points in edges.items():
            if not isinstance(points, list) or len(points) != 2:
                continue
            p1 = as_point(points[0])
            p2 = as_point(points[1])
            if not p1 or not p2:
                continue
            p1 = transform_point(p1, image_height, coord_system)
            p2 = transform_point(p2, image_height, coord_system)
            draw.line((p1[0], p1[1], p2[0], p2[1]), fill=rgba, width=2)
            draw.ellipse(
                (p1[0] - 2, p1[1] - 2, p1[0] + 2, p1[1] + 2),
                fill=rgba,
            )
            draw.ellipse(
                (p2[0] - 2, p2[1] - 2, p2[0] + 2, p2[1] + 2),
                fill=rgba,
            )
            if show_net_labels and not first_label_done:
                draw_label(
                    draw,
                    (int((p1[0] + p2[0]) / 2) + 3, int((p1[1] + p2[1]) / 2) + 3),
                    str(net_name),
                    font,
                    color,
                )
                first_label_done = True
            count += 1
    return count


def draw_header(
    overlay: Image.Image,
    case_name: str,
    counts: Dict[str, int],
    font: ImageFont.ImageFont,
) -> None:
    draw = ImageDraw.Draw(overlay, "RGBA")
    text = (
        f"{case_name} | components={counts['components']} "
        f"pins={counts['pins']} edges={counts['edges']}"
    )
    bbox = draw.textbbox((8, 8), text, font=font)
    draw.rounded_rectangle(
        (bbox[0] - 4, bbox[1] - 4, bbox[2] + 4, bbox[3] + 4),
        radius=4,
        fill=(255, 255, 255, 220),
    )
    draw.text((8, 8), text, font=font, fill=(0, 0, 0, 255))


def visualize_case(
    case_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    font: ImageFont.ImageFont,
) -> Dict[str, Any]:
    image_path, json_path = find_pair(case_dir)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    base = Image.open(image_path).convert("RGB")
    overlay = base.convert("RGBA")

    counts = {"components": 0, "pins": 0, "edges": 0}
    if not args.no_nets:
        counts["edges"] = draw_nets(
            overlay, data, font, args.net_labels, args.coord_system
        )
    if not args.no_components:
        counts["components"] = draw_components(
            overlay, data, font, args.show_values, args.coord_system
        )
    if not args.no_pins:
        counts["pins"] = draw_pins(
            overlay, data, font, args.pin_labels, args.coord_system
        )

    if not args.no_header:
        draw_header(overlay, case_dir.name, counts, font)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case_dir.name}_annotated.png"
    overlay.convert("RGB").save(output_path)

    return {
        "case": case_dir.name,
        "image": str(image_path),
        "json": str(json_path),
        "output": str(output_path),
        **counts,
    }


def write_index(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    index_path = output_dir / "annotation_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw components, pins, and net edges from PCB annotation JSON files."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root containing 200_train_cases. Default: auto-detect from cwd.",
    )
    parser.add_argument(
        "--subset",
        default="200_train_cases",
        help="Subset folder name, e.g. 200_train_cases or 10_GTcase.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "annotation_preview",
        help="Directory for rendered annotation images.",
    )
    parser.add_argument("--case", default=None, help="Only visualize one case, e.g. 0001.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of cases.")
    parser.add_argument("--font-size", type=int, default=12, help="Overlay label font size.")
    parser.add_argument(
        "--coord-system",
        choices=["bottom-left", "top-left"],
        default="bottom-left",
        help=(
            "Annotation coordinate origin. The contest JSON uses bottom-left; "
            "PIL images use top-left."
        ),
    )
    parser.add_argument("--pin-labels", action="store_true", help="Show labels for pin points.")
    parser.add_argument("--net-labels", action="store_true", help="Show net names near edges.")
    parser.add_argument("--show-values", action="store_true", help="Show component Name/value.")
    parser.add_argument("--no-components", action="store_true", help="Do not draw component boxes.")
    parser.add_argument("--no-pins", action="store_true", help="Do not draw pin points.")
    parser.add_argument("--no-nets", action="store_true", help="Do not draw net edges.")
    parser.add_argument("--no-header", action="store_true", help="Do not draw summary header.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or find_dataset_root(Path.cwd())
    subset_dir = dataset_root / args.subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    output_dir = args.output_dir / args.subset
    font = load_font(args.font_size)

    rows: List[Dict[str, Any]] = []
    for idx, case_dir in enumerate(iter_cases(subset_dir, args.case), start=1):
        if args.max_cases is not None and idx > args.max_cases:
            break
        try:
            row = visualize_case(case_dir, output_dir, args, font)
            rows.append(row)
            print(
                f"[OK] {case_dir.name}: components={row['components']} "
                f"pins={row['pins']} edges={row['edges']} -> {row['output']}"
            )
        except Exception as exc:
            print(f"[FAIL] {case_dir.name}: {exc}")

    write_index(rows, output_dir)
    print(f"Done. Rendered {len(rows)} case(s) into: {output_dir}")


if __name__ == "__main__":
    main()
