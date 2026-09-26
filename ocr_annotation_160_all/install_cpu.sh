#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$ROOT/.venv"
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT/requirements-cpu.txt"
echo '安装完成，请运行 ./start_cpu.sh'
