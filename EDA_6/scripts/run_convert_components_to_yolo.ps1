Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Convert 200_train_cases component bbox/type annotations to YOLO format.
# Output: outputs\yolo_components

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

python ".\scripts\convert_components_to_yolo.py" `
  --subset "200_train_cases" `
  --class-mode "all" `
  --val-ratio 0.2 `
  --seed 42 `
  --overwrite

Write-Host ""
Write-Host "[DONE] YOLO dataset is in:"
Write-Host (Join-Path $ProjectRoot "outputs\yolo_components")
