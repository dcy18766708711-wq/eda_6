#!/usr/bin/env python3
"""
Convert contest-6 PCB component annotations to a YOLO detection dataset.

Input:
    200_train_cases/<case>/*.png
    200_train_cases/<case>/*_target_named.json

Output:
    outputs/yolo_components/
        images/train/*.png
        images/val/*.png
        labels/train/*.txt
        labels/val/*.txt
        data.yaml
        classes.json
        conversion_index.csv

Coordinate contract:
    The contest JSON uses a bottom-left origin: x increases rightward and y
    increases upward. YOLO labels use normal image coordinates with a top-left
    origin, normalized to [0, 1]. This script performs that conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


CURATED_CLASSES = [
    "box",
    "r",
    "c",
    "gnd",
    "v",
    "net_input",
    "net_output",
    "net_short",
    "pin",
    "d",
    "led",
    "switch",
    "l",
    "mosfet",
    "bjt",
    "other",
]


def find_dataset_root(start: Path) -> Path:
    """Find the directory that contains 200_train_cases."""
    candidates = [start, *start.iterdir()]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "200_train_cases").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find a dataset root containing '200_train_cases'. "
        "Pass it with --dataset-root."
    )


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


def load_cases(subset_dir: Path) -> List[Path]:
    cases = sorted(p for p in subset_dir.iterdir() if p.is_dir())
    if not cases:
        raise FileNotFoundError(f"No case folders found in {subset_dir}")
    return cases


def collect_component_types(cases: Iterable[Path]) -> Counter:
    counter: Counter = Counter()
    for case_dir in cases:
        _, json_path = find_pair(case_dir)
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for component in (data.get("components") or {}).values():
            component_type = str((component or {}).get("type") or "other")
            counter[component_type] += 1
    return counter


def build_class_list(type_counter: Counter, class_mode: str) -> List[str]:
    if class_mode == "curated":
        return CURATED_CLASSES.copy()
    if class_mode == "all":
        return [name for name, _ in type_counter.most_common()]
    raise ValueError(f"Unsupported class mode: {class_mode}")


def split_cases(
    cases: List[Path],
    val_ratio: float,
    seed: int,
    explicit_val_cases: Optional[List[str]] = None,
) -> Tuple[List[Path], List[Path]]:
    by_name = {p.name: p for p in cases}
    if explicit_val_cases:
        val_names = set(explicit_val_cases)
        missing = sorted(val_names - set(by_name))
        if missing:
            raise FileNotFoundError(f"Val case(s) not found: {missing}")
        val = [by_name[name] for name in sorted(val_names)]
        train = [p for p in cases if p.name not in val_names]
        return train, val

    shuffled = cases.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val = sorted(shuffled[:val_count], key=lambda p: p.name)
    train = sorted(shuffled[val_count:], key=lambda p: p.name)
    return train, val


def json_bbox_to_yolo(
    bbox: List[Any],
    image_width: int,
    image_height: int,
    min_box_size: float,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert [left,bottom,right,top] JSON bbox to normalized YOLO xywh."""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None

    left = min(x1, x2)
    right = max(x1, x2)
    bottom_json = min(y1, y2)
    top_json = max(y1, y2)

    # JSON bottom-left origin -> image top-left origin.
    top = image_height - top_json
    bottom = image_height - bottom_json

    # Clamp to valid image coordinates.
    left = max(0.0, min(float(image_width), left))
    right = max(0.0, min(float(image_width), right))
    top = max(0.0, min(float(image_height), top))
    bottom = max(0.0, min(float(image_height), bottom))

    box_w = right - left
    box_h = bottom - top
    if box_w < min_box_size or box_h < min_box_size:
        return None

    x_center = (left + right) / 2.0 / image_width
    y_center = (top + bottom) / 2.0 / image_height
    norm_w = box_w / image_width
    norm_h = box_h / image_height

    return x_center, y_center, norm_w, norm_h


def safe_type(component_type: Any, class_to_id: Dict[str, int], class_mode: str) -> str:
    component_type = str(component_type or "other")
    if component_type in class_to_id:
        return component_type
    if class_mode == "curated":
        return "other"
    return component_type


def prepare_output_dirs(output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_case(
    case_dir: Path,
    split: str,
    output_root: Path,
    class_to_id: Dict[str, int],
    class_mode: str,
    min_box_size: float,
) -> Dict[str, Any]:
    image_path, json_path = find_pair(case_dir)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    output_stem = case_dir.name
    output_image = output_root / "images" / split / f"{output_stem}{image_path.suffix.lower()}"
    output_label = output_root / "labels" / split / f"{output_stem}.txt"
    shutil.copy2(image_path, output_image)

    label_lines: List[str] = []
    skipped = 0
    type_counter: Counter = Counter()
    for component_key, component in (data.get("components") or {}).items():
        component_type = safe_type(component.get("type"), class_to_id, class_mode)
        class_id = class_to_id[component_type]
        yolo_box = json_bbox_to_yolo(
            component.get("bbox"), image_width, image_height, min_box_size
        )
        if yolo_box is None:
            skipped += 1
            continue
        type_counter[component_type] += 1
        x, y, w, h = yolo_box
        label_lines.append(f"{class_id} {x:.8f} {y:.8f} {w:.8f} {h:.8f}")

    output_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    return {
        "case": case_dir.name,
        "split": split,
        "image": str(image_path),
        "json": str(json_path),
        "output_image": str(output_image),
        "output_label": str(output_label),
        "width": image_width,
        "height": image_height,
        "labels": len(label_lines),
        "skipped": skipped,
        "type_counts": dict(type_counter),
    }


def write_data_yaml(output_root: Path, class_names: List[str]) -> None:
    dataset_path = output_root.resolve().as_posix()
    lines = [
        f"path: {dataset_path}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        safe_name = str(name).replace('"', '\\"')
        lines.append(f'  {idx}: "{safe_name}"')
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_class_json(output_root: Path, class_names: List[str], type_counter: Counter) -> None:
    payload = {
        "classes": {str(i): name for i, name in enumerate(class_names)},
        "class_to_id": {name: i for i, name in enumerate(class_names)},
        "source_type_counts": dict(type_counter),
        "note": (
            "Contest JSON coordinates are bottom-left origin. Labels in this "
            "YOLO dataset are converted to top-left-origin normalized xywh."
        ),
    }
    (output_root / "classes.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_index(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    index_path = output_root / "conversion_index.csv"
    fields = [
        "case",
        "split",
        "image",
        "json",
        "output_image",
        "output_label",
        "width",
        "height",
        "labels",
        "skipped",
    ]
    with index_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def parse_case_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = [part.strip() for part in value.replace(";", ",").split(",")]
    return [part for part in parts if part]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert 200_train_cases component bbox/type annotations to YOLO format."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root containing 200_train_cases. Default: auto-detect from cwd.",
    )
    parser.add_argument("--subset", default="200_train_cases", help="Input subset folder.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "yolo_components",
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--class-mode",
        choices=["all", "curated"],
        default="all",
        help=(
            "all: every observed component type gets its own class, matching the "
            "contest annotations; curated: common classes plus other, for explicit "
            "ablation only."
        ),
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument(
        "--val-cases",
        default=None,
        help="Optional comma-separated validation case ids, e.g. 0014,0030,0100.",
    )
    parser.add_argument(
        "--min-box-size",
        type=float,
        default=1.0,
        help="Skip boxes narrower/shorter than this many pixels after conversion.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before writing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or find_dataset_root(Path.cwd())
    subset_dir = dataset_root / args.subset
    cases = load_cases(subset_dir)

    type_counter = collect_component_types(cases)
    class_names = build_class_list(type_counter, args.class_mode)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    explicit_val_cases = parse_case_list(args.val_cases)
    train_cases, val_cases = split_cases(
        cases, args.val_ratio, args.seed, explicit_val_cases=explicit_val_cases
    )

    output_root = args.output_root
    prepare_output_dirs(output_root, args.overwrite)

    rows: List[Dict[str, Any]] = []
    aggregate_counter: Counter = Counter()
    for split, split_cases_iter in (("train", train_cases), ("val", val_cases)):
        for case_dir in split_cases_iter:
            row = convert_case(
                case_dir,
                split,
                output_root,
                class_to_id,
                args.class_mode,
                args.min_box_size,
            )
            rows.append(row)
            aggregate_counter.update(row["type_counts"])
            print(
                f"[OK] {case_dir.name} -> {split}, labels={row['labels']}, "
                f"skipped={row['skipped']}"
            )

    write_data_yaml(output_root, class_names)
    write_class_json(output_root, class_names, type_counter)
    write_index(output_root, rows)

    print("")
    print(f"Dataset root: {dataset_root}")
    print(f"Output root:  {output_root}")
    print(f"Train cases:  {len(train_cases)}")
    print(f"Val cases:    {len(val_cases)}")
    print(f"Classes:      {len(class_names)}")
    print(f"Labels:       {sum(row['labels'] for row in rows)}")
    print(f"Skipped:      {sum(row['skipped'] for row in rows)}")
    print(f"data.yaml:    {output_root / 'data.yaml'}")
    print("Class counts after mapping:")
    for name, count in aggregate_counter.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
