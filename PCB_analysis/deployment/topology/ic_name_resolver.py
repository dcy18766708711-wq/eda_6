#!/usr/bin/env python3
"""Resolve IC model names from component-local OCR and the IC library index.

This stage deliberately resolves only the IC identity. It does not create pin
coordinates and it does not ask the language model to invent a library part.
Uncertain results are marked ``needs_api`` for the optional Qwen correction
stage instead of being silently accepted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.coordinate import top_left_bbox_to_bottom_left  # noqa: E402
from ic_library.index import ICLibraryIndex, load_index, normalize_name  # noqa: E402


# Component references are not IC model names.  In particular, TP2 was being
# treated as an IC query for an unresolved box because the old expression only
# excluded U?/IC? references.
DESIGNATOR_RE = re.compile(
    r"^(?:U|IC|USB|JP|CN|J|TP|R|C|L|D|LED|SW|S|Q|F|X|Y|RV|VR)\??\d*[A-Z]?$",
    re.IGNORECASE,
)
VALUE_RE = re.compile(
    r"^[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?"
    r"(?:[kKmMgGuUnNpPμ渭碌惟RrFfVvAaHhWwSs%]+)?$"
)
 # Override the legacy mojibake-tolerant pattern with a Unicode-safe value
 # parser.  Keeping the assignment local makes old deployment files readable
 # while correctly filtering values such as ``1 MΩ`` and ``10 kΩ``.
VALUE_RE = re.compile(
    r"^[+\-]?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?\s*"
    r"(?:[kKmMgGuUnNpPμµΩohmOHMrFfVvAaHhWwSs%]+)?$"
)
PINLIKE_RE = re.compile(
    r"^(?:AOUT|BOUT|MODE|CP\d*|VCP|VMB?|VMIA|BVREF|V3P3(?:OUT)?|"
    r"ISEN[AB]?|STEP|DIR|DECAY|N(?:HOME|ENBL|FAULT|RESET)|"
    r"(?:AN|AO|BO|IO|IN|OUT)\d+)$",
    re.IGNORECASE,
)

IGNORE_TEXT = {
    "vcc", "vdd", "vss", "gnd", "agnd", "pgnd", "dgnd",
    "vin", "vout", "vref", "scl", "sda", "miso", "mosi",
    "reset", "nrst", "clk", "io", "nc", "no", "com",
}


def normalize_ocr_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).strip()


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return intersection / max(1e-9, area_left + area_right - intersection)


def point_to_bbox_distance(point: tuple[float, float], bbox: list[float]) -> float:
    x, y = point
    dx = max(float(bbox[0]) - x, 0.0, x - float(bbox[2]))
    dy = max(float(bbox[1]) - y, 0.0, y - float(bbox[3]))
    return float((dx * dx + dy * dy) ** 0.5)


def text_bbox(item: dict[str, Any]) -> list[float] | None:
    bbox = item.get("bbox_xyxy_top_left") or item.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None


def text_center(item: dict[str, Any]) -> tuple[float, float] | None:
    bbox = text_bbox(item)
    if bbox is None:
        return None
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def point_to_detection_distance(point: tuple[float, float], detection: dict[str, Any]) -> float:
    bbox = detection.get("bbox_xyxy_top_left")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return float("inf")
    return point_to_bbox_distance(point, [float(value) for value in bbox])


def model_text_owner(
    payload: dict[str, Any],
    text_index: int,
    text_item: dict[str, Any],
) -> int | None:
    """Return the closest linked IC detection for one model OCR box.

    ``component_text_links`` is intentionally a candidate graph and can link
    one nearby model string to several ICs.  Model identity is different from
    ordinary context text: one visible model string may belong to only one
    component.  The closest linked IC owns it; other ICs must remain
    unresolved instead of reusing the text.
    """
    detections = payload.get("yolo", {}).get("detections", [])
    center = text_center(text_item)
    if center is None:
        return None
    boxes = [
        index
        for index, detection in enumerate(detections)
        if str(detection.get("class_name", "")).casefold() == "box"
        and isinstance(detection.get("bbox_xyxy_top_left"), list)
    ]
    if not boxes:
        return None
    return min(boxes, key=lambda index: point_to_detection_distance(center, detections[index]))


def is_model_query(text: str) -> bool:
    compact = normalize_ocr_text(text)
    if len(compact) < 3:
        return False
    normalized = normalize_name(compact)
    if not normalized or normalized in IGNORE_TEXT:
        return False
    # OCR artifacts containing Chinese glyphs are usually units, captions or
    # unreadable pin labels, not part numbers in the contest library.
    if re.search(r"[\u3400-\u9fff]", compact):
        return False
    if DESIGNATOR_RE.fullmatch(compact):
        return False
    if VALUE_RE.fullmatch(compact):
        return False
    if PINLIKE_RE.fullmatch(compact):
        return False
    # Pin names such as PB6/I2C1_SCL/TIM4_CH1 are context, not IC model names.
    if "/" in compact or "_" in compact:
        return False
    # Most schematic IC values contain at least one digit. Pure words such as
    # GND/RIN/PGNDR are pin or net labels and are excluded here.
    return any(char.isdigit() for char in compact)


def model_query_variants(text: str) -> list[str]:
    """Generate bounded, library-safe OCR corrections for model identifiers."""
    value = str(text or "").strip()
    variants = [value]
    # PP-OCR frequently reads the Q suffix in DRV8824-Q1 as O1. Restrict the
    # correction to this model family so ordinary pin labels are untouched.
    corrected = re.sub(
        r"^(DRV\d+[-_]?)O(\d+[A-Z]?)$",
        r"\1Q\2",
        value,
        flags=re.IGNORECASE,
    )
    if corrected.casefold() != value.casefold():
        variants.append(corrected)
    # A frequent compact-font artefact is an inserted V/Y between P and W
    # (for example TXB0108PVWR instead of TXB0108PWR).  This rule is limited
    # to a terminal package suffix and remains subject to exact library match.
    for base in list(variants):
        suffix_corrected = re.sub(r"P[VY]W(?=R?$)", "PW", base, flags=re.IGNORECASE)
        if suffix_corrected.casefold() != base.casefold():
            variants.append(suffix_corrected)
    # One-character substitutions cover the common PCB-font mistakes without
    # allowing a free-form language-model rewrite.  Each resulting string is
    # still required to match a local library candidate downstream.
    confusions = {
        "0": "OQ",
        "O": "0Q",
        "Q": "0O",
        "1": "IL",
        "I": "1L",
        "L": "1I",
        "5": "S",
        "S": "5",
        "8": "B",
        "B": "8",
    }
    for base in list(variants):
        upper = base.upper()
        for position, char in enumerate(upper):
            if char not in confusions:
                continue
            for replacement in confusions[char]:
                candidate = upper[:position] + replacement + upper[position + 1 :]
                if candidate.casefold() != value.casefold():
                    variants.append(candidate)
    # Deterministic deduplication protects the API budget on long OCR text.
    return list(dict.fromkeys(variants))[:24]


def linked_text_indices(payload: dict[str, Any], detection_index: int) -> set[int]:
    links = payload.get("component_text_links", [])
    if not isinstance(links, list) or detection_index >= len(links):
        return set()
    result: set[int] = set()
    for item in links[detection_index].get("text_candidates", []):
        try:
            result.add(int(item["text_index"]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def match_detection(
    component: dict[str, Any],
    detections: list[dict[str, Any]],
    image_height: float,
) -> tuple[int | None, float]:
    component_bbox = component.get("bbox")
    if not isinstance(component_bbox, list) or len(component_bbox) != 4:
        return None, 0.0
    best_index: int | None = None
    best_score = 0.0
    for index, detection in enumerate(detections):
        bbox = detection.get("bbox_xyxy_top_left")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        detection_bbox = top_left_bbox_to_bottom_left(bbox, image_height)
        score = bbox_iou(component_bbox, detection_bbox)
        if score > best_score:
            best_index, best_score = index, score
    return best_index, best_score


def collect_component_texts(
    component: dict[str, Any],
    detection_index: int | None,
    payload: dict[str, Any],
    image_height: float,
    *,
    context_margin: float,
    local_ocr_items: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    texts = payload.get("ocr", {}).get("texts", [])
    if not isinstance(texts, list):
        return []
    component_bbox_bl = component.get("bbox")
    if not isinstance(component_bbox_bl, list) or len(component_bbox_bl) != 4:
        return []
    component_bbox = [
        float(component_bbox_bl[0]),
        image_height - float(component_bbox_bl[3]),
        float(component_bbox_bl[2]),
        image_height - float(component_bbox_bl[1]),
    ]
    linked = linked_text_indices(payload, detection_index) if detection_index is not None else set()
    candidates: list[dict[str, Any]] = []
    candidates_source: list[tuple[int, dict[str, Any], bool, str]] = []
    for index, item in enumerate(texts):
        if isinstance(item, dict):
            candidates_source.append((index, item, index in linked, "page_ocr"))
    for index, item in enumerate(local_ocr_items or []):
        if isinstance(item, dict):
            candidates_source.append((index, item, False, "ic_local_ocr"))

    for index, item, is_linked, source in candidates_source:
        if not isinstance(item, dict):
            continue
        bbox = text_bbox(item)
        center = text_center(item)
        if bbox is None or center is None:
            continue
        distance = point_to_bbox_distance(center, component_bbox)
        if not is_linked and distance > context_margin:
            continue
        text = normalize_ocr_text(item.get("text"))
        if not is_model_query(text):
            continue
        if source == "page_ocr":
            owner = model_text_owner(payload, index, item)
            if owner is not None and owner != detection_index:
                # This model string is already owned by a closer IC.  It is
                # not valid evidence for the current component.
                continue
        association = 1.0 if is_linked else max(0.05, 1.0 - distance / max(1.0, context_margin))
        candidates.append(
            {
                "text_index": index,
                "text": item.get("text", ""),
                "normalized_text": text,
                "ocr_score": round(float(item.get("score", 0.0) or 0.0), 6),
                "association_score": round(association, 6),
                "distance_px": round(distance, 2),
                "linked": is_linked,
                "source": source,
                "bbox": bbox,
            }
        )
    return sorted(candidates, key=lambda item: (-item["association_score"], -item["ocr_score"]))


def merge_model_text_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join adjacent OCR fragments only when they form one model-name line."""
    merged: list[dict[str, Any]] = []
    for left_index, left in enumerate(items):
        left_box = left.get("bbox")
        if not isinstance(left_box, list) or len(left_box) != 4:
            continue
        left_center_y = (float(left_box[1]) + float(left_box[3])) / 2.0
        left_height = max(1.0, float(left_box[3]) - float(left_box[1]))
        for right in items[left_index + 1 :]:
            right_box = right.get("bbox")
            if not isinstance(right_box, list) or len(right_box) != 4:
                continue
            right_center_y = (float(right_box[1]) + float(right_box[3])) / 2.0
            right_height = max(1.0, float(right_box[3]) - float(right_box[1]))
            if abs(left_center_y - right_center_y) > max(10.0, 0.65 * max(left_height, right_height)):
                continue
            first, second = (left, right) if float(left_box[0]) <= float(right_box[0]) else (right, left)
            first_box, second_box = first["bbox"], second["bbox"]
            gap = float(second_box[0]) - float(first_box[2])
            if gap < -2.0 or gap > max(30.0, 1.5 * max(left_height, right_height)):
                continue
            text = normalize_ocr_text(first["text"]) + normalize_ocr_text(second["text"])
            if not is_model_query(text):
                continue
            merged.append(
                {
                    "text_index": None,
                    "text": text,
                    "normalized_text": text,
                    "ocr_score": round(min(float(first["ocr_score"]), float(second["ocr_score"])), 6),
                    "association_score": round(
                        min(float(first["association_score"]), float(second["association_score"])), 6
                    ),
                    "distance_px": round(
                        min(float(first["distance_px"]), float(second["distance_px"])), 2
                    ),
                    "linked": bool(first["linked"] and second["linked"]),
                    "source": "merged_model_text",
                    "source_text_indices": [first.get("text_index"), second.get("text_index")],
                    "bbox": [
                        min(float(first_box[0]), float(second_box[0])),
                        min(float(first_box[1]), float(second_box[1])),
                        max(float(first_box[2]), float(second_box[2])),
                        max(float(first_box[3]), float(second_box[3])),
                    ],
                }
            )
    unique: dict[str, dict[str, Any]] = {}
    for item in merged:
        key = normalize_name(item["text"])
        if key and (key not in unique or item["ocr_score"] > unique[key]["ocr_score"]):
            unique[key] = item
    return list(unique.values())


def resolve_component(
    component: dict[str, Any],
    payload: dict[str, Any],
    index: ICLibraryIndex,
    image_height: float,
    *,
    context_margin: float = 100.0,
    max_candidates: int = 5,
    local_ocr_items: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    detections = payload.get("yolo", {}).get("detections", [])
    detection_index, bbox_score = match_detection(component, detections, image_height)
    text_candidates = collect_component_texts(
        component,
        detection_index,
        payload,
        image_height,
        context_margin=context_margin,
        local_ocr_items=local_ocr_items,
    )
    text_candidates = text_candidates + merge_model_text_candidates(text_candidates)

    aggregate: dict[str, dict[str, Any]] = {}
    query_evidence: list[dict[str, Any]] = []
    for text_item in text_candidates:
        variant_candidates: dict[str, dict[str, Any]] = {}
        for query_variant in model_query_variants(text_item["normalized_text"]):
            for candidate in index.find_candidates(query_variant, limit=max_candidates):
                name = str(candidate["name"])
                current = variant_candidates.get(name)
                if current is None or float(candidate.get("score", 0.0)) > float(current.get("score", 0.0)):
                    variant_candidates[name] = candidate
        library_candidates = sorted(
            variant_candidates.values(),
            key=lambda item: (-float(item.get("score", 0.0)), str(item.get("name", ""))),
        )[:max_candidates]
        library_candidates = [item for item in library_candidates if item.get("score", 0.0) >= 0.45]
        if not library_candidates:
            continue
        for rank, candidate in enumerate(library_candidates):
            # Keep second/third candidates: one wrong OCR character should
            # not make the top result the only possible model.
            rank_factor = 1.0 / (1.0 + 0.08 * rank)
            evidence = (
                float(candidate.get("score", 0.0))
                * max(0.1, float(text_item["ocr_score"]))
                * max(0.1, float(text_item["association_score"]))
                * rank_factor
            )
            name = str(candidate["name"])
            current = aggregate.get(name)
            if current is None or evidence > current["evidence"]:
                aggregate[name] = {
                    "name": name,
                    "evidence": evidence,
                    "match_score": candidate.get("score", 0.0),
                    "match": candidate.get("match"),
                    "matched_alias": candidate.get("matched_alias", ""),
                    "physical_pin_count": candidate.get("physical_pin_count"),
                    "pin_numbers": candidate.get("pin_numbers", []),
                    "query": text_item["text"],
                    "text_index": text_item["text_index"],
                }
        query_evidence.append(
            {
                "query": text_item["text"],
                "text_index": text_item["text_index"],
                "ocr_score": text_item["ocr_score"],
                "association_score": text_item["association_score"],
                "candidates": library_candidates,
            }
        )

    ranked = sorted(aggregate.values(), key=lambda item: (-item["evidence"], -float(item["match_score"]), item["name"]))
    best = ranked[0] if ranked else None
    second_score = float(ranked[1]["evidence"]) if len(ranked) > 1 else 0.0
    best_score = float(best["evidence"]) if best else 0.0
    margin = best_score - second_score
    match_score = float(best["match_score"]) if best else 0.0
    accepted = bool(best and match_score >= 0.90 and (margin >= 0.05 or match_score >= 0.99))

    return {
        "model_name": best["name"] if accepted else None,
        "status": "matched" if accepted else ("needs_api" if ranked else "unresolved"),
        "needs_api": not accepted,
        "score": round(best_score, 6),
        "match_score": round(match_score, 6),
        "margin": round(margin, 6),
        "bbox_match_iou": round(bbox_score, 6),
        "detection_index": detection_index,
        "query": best["query"] if best else None,
        "physical_pin_count": best.get("physical_pin_count") if accepted else None,
        "pin_numbers": best.get("pin_numbers", []) if accepted else [],
        "candidates": ranked[:max_candidates],
        "ocr_evidence": query_evidence,
        "ocr_texts": text_candidates,
        "reason": (
            "accepted exact or high-confidence IC library match"
            if accepted
            else "no reliable IC library match; send local crop and candidates to optional Qwen stage"
        ),
    }


def resolve_all(
    payload: dict[str, Any],
    components: dict[str, Any],
    index: ICLibraryIndex,
    *,
    context_margin: float = 100.0,
    local_ocr_by_component: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    image_height = float(payload.get("image_size", {}).get("height", 0.0) or 0.0)
    if image_height <= 0:
        raise ValueError("intermediate JSON must contain image_size.height")
    output: dict[str, Any] = {}
    for component_key, component in components.items():
        if component.get("type") != "box":
            continue
        output[component_key] = resolve_component(
            component,
            payload,
            index,
            image_height,
            context_margin=context_margin,
            local_ocr_items=(local_ocr_by_component or {}).get(component_key, []),
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate", type=Path, required=True, help="YOLO+OCR intermediate JSON")
    parser.add_argument("--component-json", type=Path, required=True, help="component_builder output JSON")
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "ic_library" / "BoxICLibrary_200cases.index.json",
    )
    parser.add_argument("--output", type=Path, required=True, help="automatic IC mapping JSON")
    parser.add_argument("--context-margin", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.intermediate.read_text(encoding="utf-8"))
    component_payload = json.loads(args.component_json.read_text(encoding="utf-8"))
    index = load_index(args.index)
    result = resolve_all(
        payload,
        component_payload.get("components", {}),
        index,
        context_margin=args.context_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    matched = sum(1 for item in result.values() if item.get("status") == "matched")
    needs_api = sum(1 for item in result.values() if item.get("needs_api"))
    print(f"ICs={len(result)}, matched={matched}, needs_api={needs_api} -> {args.output}")


if __name__ == "__main__":
    main()
