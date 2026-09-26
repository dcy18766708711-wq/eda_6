#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="${PYTHON_BIN:-python3}"; fi
export CUDA_VISIBLE_DEVICES=""
export PADDLE_PDX_CACHE_HOME="$ROOT/.paddlex"
exec "$PYTHON" "$ROOT/annotate_ocr_160.py" \
  --web --host 127.0.0.1 --port "${PORT:-8766}" \
  --source "$ROOT/source" --output "$ROOT/result" \
  --ocr-device cpu --ocr-model PP-OCRv6_small_rec
