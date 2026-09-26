$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    $Python = "python"
}

$env:CUDA_VISIBLE_DEVICES = ""
$env:PADDLE_PDX_CACHE_HOME = Join-Path $Root ".paddlex"

if (-not $env:PORT) {
    $env:PORT = "8766"
}

& $Python (Join-Path $Root "annotate_ocr_160.py") `
    --web --host 127.0.0.1 --port $env:PORT `
    --source (Join-Path $Root "source") --output (Join-Path $Root "result") `
    --ocr-device cpu --ocr-model PP-OCRv6_small_rec
