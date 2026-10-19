#!/usr/bin/env python3
"""Build and query the Problem 6 IC library index.

Authority order follows the latest contest group reply:

1. ``BoxICLibrary_200cases.json`` is the primary library.
2. ``cpntLibrary.json`` is an optional compatibility fallback only.

The library supplies candidate IC names and pin metadata. It has no image
coordinates, so this module never creates pin points or component boxes.
Repeated records with the same pin number are retained as alternatives; the
number of physical pins is the count of distinct ``number`` values, not the
length of the raw ``pins`` array.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_LIBRARY = ROOT / "ic_library" / "BoxICLibrary_200cases.json"
FALLBACK_LIBRARY = ROOT / "ic_library" / "cpntLibrary.json"


def normalize_name(value: Any) -> str:
    """Normalize a lookup key without changing the original library value."""
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    text = text.replace("μ", "u").replace("µ", "u")
    return re.sub(r"[^a-z0-9]+", "", text)


def display_name_variants(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    variants = {text, Path(text).stem}
    return {item for item in variants if item}


def natural_pin_key(number: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", str(number))
    return tuple(int(part) if part.isdigit() else part.casefold() for part in parts)


def _unique_sorted(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)}, key=natural_pin_key)


def _clean_pin(record: Any) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError("pin record must be an object")
    number = str(record.get("number", "")).strip()
    if not number:
        raise ValueError("pin record has no number")
    return {
        "number": number,
        "name": str(record.get("name", "")).strip(),
        "pin_type": str(record.get("pin_type", "")).strip(),
        "pin_id": str(record.get("pin_id", "")).strip(),
    }


def normalize_record(canonical_name: str, record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"library record is not an object: {canonical_name}")
    raw_pins = record.get("pins", [])
    if not isinstance(raw_pins, list):
        raise ValueError(f"pins is not an array: {canonical_name}")

    pins_by_number: dict[str, list[dict[str, str]]] = {}
    invalid_pin_records = 0
    for raw_pin in raw_pins:
        try:
            pin = _clean_pin(raw_pin)
        except ValueError:
            invalid_pin_records += 1
            continue
        pins_by_number.setdefault(pin["number"], []).append(pin)

    for number, records in pins_by_number.items():
        unique_records = {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item for item in records
        }
        pins_by_number[number] = sorted(
            unique_records.values(),
            key=lambda item: (item["name"].casefold(), item["pin_type"], item["pin_id"]),
        )

    aliases: set[str] = set()
    aliases.update(display_name_variants(canonical_name))
    aliases.update(display_name_variants(record.get("esym_path", "")))
    same_hash = record.get("same_hash_components", [])
    if isinstance(same_hash, list):
        for item in same_hash:
            if isinstance(item, dict):
                aliases.update(display_name_variants(item.get("title", "")))

    return {
        "name": canonical_name,
        "esym_path": str(record.get("esym_path", "")),
        "aliases": sorted(aliases, key=str.casefold),
        "pin_numbers": _unique_sorted(pins_by_number.keys()),
        "physical_pin_count": len(pins_by_number),
        "pins_by_number": {
            number: pins_by_number[number]
            for number in sorted(pins_by_number, key=natural_pin_key)
        },
        "same_hash_components": same_hash if isinstance(same_hash, list) else [],
        "invalid_pin_records": invalid_pin_records,
    }


def read_library(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"IC library root must be an object: {path}")
    return payload


def build_index(library_path: Path, *, authority: str | None = None) -> dict[str, Any]:
    """Build a serializable index from one library JSON file."""
    raw = read_library(library_path)
    records: dict[str, dict[str, Any]] = {}
    alias_index: dict[str, list[str]] = {}
    invalid_records: list[str] = []
    for canonical_name, raw_record in raw.items():
        try:
            normalized = normalize_record(str(canonical_name), raw_record)
        except ValueError:
            invalid_records.append(str(canonical_name))
            continue
        records[normalized["name"]] = normalized
        for alias in normalized["aliases"]:
            key = normalize_name(alias)
            if key:
                alias_index.setdefault(key, []).append(normalized["name"])

    for key in alias_index:
        alias_index[key] = sorted(set(alias_index[key]), key=str.casefold)

    return {
        "format": "eda6_ic_library_index_v1",
        # Keep the generated index portable; never persist a developer's
        # absolute local path in a deployment artifact.
        "source": library_path.name,
        "authority": authority or library_path.name,
        "coordinates_available": False,
        "record_count": len(records),
        "pin_record_count": sum(
            sum(len(items) for items in record["pins_by_number"].values())
            for record in records.values()
        ),
        "invalid_records": invalid_records,
        "records": records,
        "alias_index": alias_index,
    }


class ICLibraryIndex:
    """Query wrapper around a built index."""

    def __init__(self, payload: dict[str, Any]):
        if payload.get("format") != "eda6_ic_library_index_v1":
            raise ValueError("unsupported IC library index format")
        self.payload = payload
        self.records: dict[str, dict[str, Any]] = payload.get("records", {})
        self.alias_index: dict[str, list[str]] = payload.get("alias_index", {})

    @property
    def source(self) -> str:
        return str(self.payload.get("source", ""))

    @property
    def authority(self) -> str:
        return str(self.payload.get("authority", ""))

    def get(self, canonical_name: str) -> dict[str, Any] | None:
        return self.records.get(canonical_name)

    def pin_numbers(self, canonical_name: str) -> list[str]:
        record = self.get(canonical_name)
        return [] if record is None else list(record["pin_numbers"])

    def pins(self, canonical_name: str, number: str | None = None) -> Any:
        record = self.get(canonical_name)
        if record is None:
            return {} if number is None else []
        if number is None:
            return record["pins_by_number"]
        return list(record["pins_by_number"].get(str(number), []))

    def find_candidates(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return ranked candidates; names remain exact library names."""
        query_text = str(query or "").strip()
        query_key = normalize_name(query_text)
        if not query_key:
            return []

        exact_names = self.alias_index.get(query_key, [])
        scored: dict[str, dict[str, Any]] = {}
        for name in exact_names:
            scored[name] = {
                "name": name,
                "score": 1.0,
                "matched_alias": query_text,
                "match": "exact_normalized",
            }

        for name, record in self.records.items():
            best_score = scored.get(name, {}).get("score", 0.0)
            best_alias = ""
            best_match = "fuzzy"
            for alias in record["aliases"]:
                alias_key = normalize_name(alias)
                if not alias_key:
                    continue
                shorter = min(len(query_key), len(alias_key))
                if shorter >= 4 and (query_key in alias_key or alias_key in query_key):
                    longer = max(len(query_key), len(alias_key))
                    score = 0.90 + 0.08 * shorter / max(1, longer)
                    match = "contains"
                else:
                    score = SequenceMatcher(None, query_key, alias_key).ratio()
                    match = "fuzzy"
                if score > best_score:
                    best_score, best_alias, best_match = score, alias, match
            if best_score >= 0.45:
                scored[name] = {
                    "name": name,
                    "score": round(float(best_score), 6),
                    "matched_alias": best_alias,
                    "match": best_match,
                }

        ranked = sorted(scored.values(), key=lambda item: (-item["score"], item["name"].casefold()))
        for item in ranked[:limit]:
            record = self.records[item["name"]]
            item["physical_pin_count"] = record["physical_pin_count"]
            item["pin_numbers"] = record["pin_numbers"]
        return ranked[:limit]


def load_index(path: str | Path) -> ICLibraryIndex:
    index_path = Path(path)
    return ICLibraryIndex(json.loads(index_path.read_text(encoding="utf-8")))


def resolve_library(primary: Path, fallback: Path | None) -> tuple[Path, str]:
    if primary.is_file():
        return primary, "BoxICLibrary_200cases.json"
    if fallback is not None and fallback.is_file():
        print(f"WARNING: primary IC library not found; using fallback {fallback}", file=sys.stderr)
        return fallback, "cpntLibrary.json_fallback"
    raise FileNotFoundError(f"No IC library found: primary={primary}, fallback={fallback}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=PRIMARY_LIBRARY)
    parser.add_argument("--fallback", type=Path, default=FALLBACK_LIBRARY)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ic_library" / "BoxICLibrary_200cases.index.json",
    )
    parser.add_argument("--query", help="Optional IC name query after building the index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library_path, authority = resolve_library(args.library, args.fallback)
    index_payload = build_index(library_path, authority=authority)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"authority={authority}, records={index_payload['record_count']}, "
        f"pin_records={index_payload['pin_record_count']} -> {args.output}"
    )
    if args.query:
        index = ICLibraryIndex(index_payload)
        print(json.dumps(index.find_candidates(args.query), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
