#!/usr/bin/env python3
"""Resolve uncertain IC identities with the configured Qwen API.

The API is a text/semantic fallback only. It may correct an OCR string or
choose among local IC-library candidates, but it cannot create coordinates or
invent a pin table that is absent from the local library.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api_config  # noqa: E402
from common.coordinate import bottom_left_bbox_to_top_left  # noqa: E402
from ic_library.index import ICLibraryIndex  # noqa: E402


DESIGNATOR_OUTPUT_RE = re.compile(
    r"^(?:USB|LED|SW|TP|JP|CN|RV|VR|IC|R|C|L|U|Q|D|F|S|X|Y|J|H)\d+[A-Z]?$",
    re.IGNORECASE,
)


SYSTEM_PROMPT = """你是 PCB 原理图 OCR 纠错助手。
你的任务是从本地 OCR 文本和候选 IC 型号中选择正确的 IC 型号。
只允许从给定 candidates 中选择，不确定时必须返回 model_name=null、uncertain=true。
不要输出 pin 坐标，不要修改组件 bbox，不要推测候选列表之外的型号。
只返回一个 JSON 对象，不要 Markdown，不要额外解释：
{"model_name": "候选型号或null", "corrected_text": "纠正后的文字", "confidence": 0.0, "uncertain": true, "reason": "简短原因"}
"""


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content or "")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Qwen response does not contain a JSON object")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Qwen response JSON must be an object")
    return value


def _crop_data_uri(
    image: np.ndarray,
    bbox_top_left: list[float],
    *,
    max_side: int,
    margin: int = 40,
) -> str | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox_top_left]
    left = max(0, int(x1 - margin))
    top = max(0, int(y1 - margin))
    right = min(width, int(x2 + margin))
    bottom = min(height, int(y2 + margin))
    if right <= left or bottom <= top:
        return None
    crop = image[top:bottom, left:right]
    largest = max(crop.shape[:2])
    if largest > max_side:
        scale = max_side / largest
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _request_payload(
    resolution: dict[str, Any],
    component_key: str,
    *,
    image_data_uri: str | None,
) -> dict[str, Any]:
    candidates = []
    for item in resolution.get("candidates", [])[:8]:
        candidates.append(
            {
                "name": item.get("name"),
                "match_score": item.get("match_score"),
                "physical_pin_count": item.get("physical_pin_count"),
            }
        )
    ocr_texts = []
    for item in resolution.get("ocr_texts", [])[:12]:
        ocr_texts.append(
            {
                "text": item.get("text"),
                "ocr_score": item.get("ocr_score"),
                "distance_px": item.get("distance_px"),
            }
        )
    prompt = {
        "task": "correct_ic_model_name",
        "component_key": component_key,
        "local_ocr_texts": ocr_texts,
        "library_candidates": candidates,
        "local_best_query": resolution.get("query"),
        "instruction": "只从 library_candidates 选择；如果图中文字和候选都不能确认，model_name 返回 null。",
    }
    text_content = json.dumps(prompt, ensure_ascii=False)
    if image_data_uri:
        user_content: Any = [
            {"type": "text", "text": text_content},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    else:
        user_content = text_content
    return {
        "model": api_config.API_MODEL,
        "temperature": float(getattr(api_config, "API_TEMPERATURE", 0.0)),
        "max_tokens": int(getattr(api_config, "API_MAX_TOKENS", 300)),
        "enable_thinking": bool(getattr(api_config, "API_ENABLE_THINKING", False)),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }


def _cache_file() -> Path:
    configured = str(getattr(api_config, "API_CACHE_PATH", "")).strip()
    return Path(configured) if configured else ROOT / "outputs" / "qwen_ic_cache.json"


def _cache_key(
    resolution: dict[str, Any],
    component_key: str,
    image_data_uri: str | None,
) -> str:
    """Hash only the semantic request inputs; never persist the API key."""
    fingerprint = {
        "model": str(api_config.API_MODEL),
        "component_key": component_key,
        "query": resolution.get("query"),
        "ocr_texts": resolution.get("ocr_texts", []),
        "candidates": [
            {"name": item.get("name"), "match_score": item.get("match_score")}
            for item in resolution.get("candidates", [])[:8]
        ],
        "image_sha256": hashlib.sha256((image_data_uri or "").encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(path: Path, key: str, result: dict[str, Any]) -> None:
    cache = _read_cache(path)
    cache[key] = {"result": result, "cached_at": int(time.time())}
    # Bound the file so a long contest run cannot accumulate unbounded OCR data.
    if len(cache) > 5000:
        cache = dict(sorted(cache.items(), key=lambda item: item[1].get("cached_at", 0))[-4000:])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _post_json(payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = str(api_config.API_ENDPOINT).strip()
    key = str(api_config.API_KEY).strip()
    if not endpoint or "YOUR_" in endpoint or endpoint.startswith("PASTE_"):
        raise RuntimeError("API_ENDPOINT is not configured")
    if not key or key.startswith("PASTE_") or key.startswith("YOUR_"):
        raise RuntimeError("API_KEY is not configured")
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(api_config.API_TIMEOUT_SECONDS)) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Qwen HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Qwen connection failed: {exc.reason}") from exc
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Qwen response is not JSON: {raw[:500]}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("Qwen response root must be an object")
    return result


def call_qwen(
    resolution: dict[str, Any],
    component_key: str,
    *,
    image: np.ndarray | None = None,
    bbox_top_left: list[float] | None = None,
) -> dict[str, Any]:
    image_data_uri = None
    if (
        bool(getattr(api_config, "API_SEND_IMAGE", True))
        and image is not None
        and bbox_top_left is not None
    ):
        image_data_uri = _crop_data_uri(
            image,
            bbox_top_left,
            max_side=int(getattr(api_config, "API_IMAGE_MAX_SIDE", 1280)),
        )
    payload = _request_payload(resolution, component_key, image_data_uri=image_data_uri)
    cache_path = _cache_file()
    cache_key = _cache_key(resolution, component_key, image_data_uri)
    cached = _read_cache(cache_path).get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("result"), dict):
        result = dict(cached["result"])
        result["cache_hit"] = True
        return result
    retries = int(getattr(api_config, "API_MAX_RETRIES", 1))
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = _post_json(payload)
            choices = response.get("choices", [])
            if not choices:
                raise RuntimeError("Qwen response has no choices")
            message = choices[0].get("message", {})
            parsed = parse_json_object(_message_text(message.get("content")))
            parsed["raw_response_id"] = response.get("id")
            parsed["image_sent"] = bool(image_data_uri)
            parsed["cache_hit"] = False
            _write_cache(cache_path, cache_key, parsed)
            return parsed
        except (RuntimeError, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0, 0.5 * (attempt + 1)))
    raise RuntimeError(str(last_error) if last_error else "Qwen request failed")


def call_qwen_designator(
    *,
    component_type: str,
    component_index: int,
    ocr_candidates: list[dict[str, Any]],
    partial_designator: str | None = None,
    image: np.ndarray | None = None,
    bbox_top_left: list[float] | None = None,
) -> dict[str, Any]:
    """Ask Qwen to read one visible designator, without asking for geometry.

    This is intentionally separate from IC model resolution.  The API may
    correct text content only; the local detector still owns the component
    bbox and the component type.  The caller must validate the returned key
    against the allowed role before promoting it.
    """
    image_data_uri = None
    if (
        bool(getattr(api_config, "API_SEND_IMAGE", True))
        and image is not None
        and bbox_top_left is not None
    ):
        image_data_uri = _crop_data_uri(
            image,
            bbox_top_left,
            max_side=int(getattr(api_config, "API_IMAGE_MAX_SIDE", 1280)),
            margin=60,
        )
    prompt = {
        "task": "read_visible_component_designator",
        "component_type": component_type,
        "component_index": component_index,
        "partial_designator": partial_designator,
        "local_ocr_candidates": ocr_candidates[:12],
        "rules": [
            "read only characters visibly present in the supplied crop",
            "do not infer a sequential number from neighboring components",
            "do not return a pin name, net label, or IC model name",
            "return null when the crop is ambiguous",
        ],
        "output": {"designator": "string or null", "confidence": 0.0, "uncertain": True},
    }
    user_content: Any = json.dumps(prompt, ensure_ascii=False)
    if image_data_uri:
        user_content = [
            {"type": "text", "text": json.dumps(prompt, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": image_data_uri}},
        ]
    payload = {
        "model": api_config.API_MODEL,
        "temperature": float(getattr(api_config, "API_TEMPERATURE", 0.0)),
        "max_tokens": int(getattr(api_config, "API_MAX_TOKENS", 120)),
        "enable_thinking": bool(getattr(api_config, "API_ENABLE_THINKING", False)),
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": "Return JSON only. You may correct visible text, never coordinates or component type.",
            },
            {"role": "user", "content": user_content},
        ],
    }
    fingerprint = {
        "kind": "designator",
        "model": str(api_config.API_MODEL),
        "component_type": component_type,
        "component_index": component_index,
        "partial_designator": partial_designator,
        "ocr_candidates": ocr_candidates[:12],
        "image_sha256": hashlib.sha256((image_data_uri or "").encode("utf-8")).hexdigest(),
    }
    cache_key = hashlib.sha256(
        json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = _cache_file()
    cached = _read_cache(cache_path).get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("result"), dict):
        result = dict(cached["result"])
        result["cache_hit"] = True
        return result
    response = _post_json(payload)
    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("Qwen designator response has no choices")
    message = choices[0].get("message", {})
    result = parse_json_object(_message_text(message.get("content")))
    result["cache_hit"] = False
    result["image_sent"] = bool(image_data_uri)
    _write_cache(cache_path, cache_key, result)
    return result


def accept_qwen_designator(result: dict[str, Any], component_type: str) -> tuple[str | None, float]:
    """Validate API text without accepting a guessed or wrong-role key."""
    if bool(result.get("uncertain", True)):
        return None, 0.0
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    # Some compatible endpoints place the requested visible text in
    # ``corrected_text`` instead of the task-specific ``designator`` field.
    # It is still safe to use as a fallback because the same strict
    # designator regex, confidence threshold, and type-prefix gate apply.
    value = re.sub(
        r"[\s_-]+",
        "",
        str(result.get("designator") or result.get("corrected_text") or ""),
    ).upper()
    if confidence < 0.85 or not DESIGNATOR_OUTPUT_RE.fullmatch(value):
        return None, confidence
    prefix = re.match(r"[A-Z]+", value)
    allowed = {
        # ``box`` is the detector class for rectangular ICs and rectangular
        # connectors in the public data.  A connector can visibly use H1/H2;
        # this is accepted only here, after the API has reviewed the local
        # crop.  The deterministic full-page assignment remains type-gated
        # and does not globally treat every H token as a box component.
        "box": {"U", "IC", "USB", "JP", "CN", "J", "H"},
        "amp": {"U", "IC"},
        "r": {"R"},
        "c": {"C"},
        "l": {"L"},
        "d": {"D"},
        "led": {"LED"},
        "switch": {"SW", "S"},
        "fuse": {"F"},
        "crystal": {"X", "Y"},
        "bjt": {"Q", "U"},
        "mosfet": {"Q", "U"},
        "testpoint": {"TP"},
        "pin": {"TP"},
        "circle_header": {"H"},
        "other": {"H"},
    }.get(component_type, set())
    if not prefix or prefix.group(0) not in allowed:
        return None, confidence
    return value, confidence


def _accept_api_result(
    result: dict[str, Any],
    index: ICLibraryIndex,
    allowed_names: set[str],
) -> tuple[str | None, dict[str, Any] | None]:
    if bool(result.get("uncertain", True)):
        return None, None
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.70:
        return None, None
    requested = str(result.get("model_name") or result.get("corrected_text") or "").strip()
    if not requested:
        return None, None
    candidates = index.find_candidates(requested, limit=1)
    if not candidates or float(candidates[0].get("score", 0.0)) < 0.99:
        return None, None
    canonical = str(candidates[0]["name"])
    if canonical not in allowed_names:
        return None, None
    return canonical, index.get(canonical)


def resolve_needs_api(
    resolutions: dict[str, dict[str, Any]],
    components: dict[str, Any],
    image: np.ndarray,
    image_height: float,
    index: ICLibraryIndex,
) -> dict[str, dict[str, Any]]:
    """Update local resolutions with accepted Qwen results."""
    updated = {key: dict(value) for key, value in resolutions.items()}
    # Evaluation and geometry debugging must be reproducible without network
    # latency.  The environment switch leaves the contest API configuration
    # untouched and disables only this invocation.
    if os.getenv("EDA6_DISABLE_API", "").strip().lower() in {"1", "true", "yes"}:
        return updated
    if not bool(getattr(api_config, "API_ENABLED", False)):
        return updated
    limit = int(getattr(api_config, "API_MAX_REQUESTS_PER_IMAGE", 30))
    attempted = 0
    for component_key, resolution in updated.items():
        if not resolution.get("needs_api") or attempted >= limit:
            continue
        component = components.get(component_key, {})
        bbox = component.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            resolution["api_attempted"] = False
            resolution["api_error"] = "component bbox is invalid"
            continue
        bbox_top_left = bottom_left_bbox_to_top_left(bbox, image_height)
        attempted += 1
        resolution["api_attempted"] = True
        try:
            api_result = call_qwen(
                resolution,
                component_key,
                image=image,
                bbox_top_left=bbox_top_left,
            )
            resolution["api_result"] = api_result
            allowed_names = {
                str(item.get("name"))
                for item in resolution.get("candidates", [])
                if item.get("name")
            }
            model_name, record = _accept_api_result(api_result, index, allowed_names)
            if model_name and record:
                resolution["model_name"] = model_name
                resolution["status"] = "matched"
                resolution["needs_api"] = False
                resolution["source"] = "qwen_api"
                resolution["physical_pin_count"] = record.get("physical_pin_count")
                resolution["pin_numbers"] = record.get("pin_numbers", [])
            else:
                resolution["api_error"] = "Qwen result was uncertain or not present in the local IC library"
        except Exception as exc:  # keep one API failure from stopping the image
            resolution["api_error"] = str(exc)
            resolution["status"] = "needs_api"
    if attempted >= limit:
        for resolution in updated.values():
            if resolution.get("needs_api") and not resolution.get("api_attempted"):
                resolution["api_error"] = "API_MAX_REQUESTS_PER_IMAGE reached"
    return updated
