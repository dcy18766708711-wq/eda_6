Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Visualize all annotations in 200_train_cases.
# Output: outputs\annotation_preview\200_train_cases

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

python ".\scripts\visualize_annotations.py" `
  --subset "200_train_cases" `
  --coord-system "bottom-left" `
  --output-dir ".\outputs\annotation_preview"

Write-Host ""
Write-Host "[DONE] Annotation previews are in:"
Write-Host (Join-Path $ProjectRoot "outputs\annotation_preview\200_train_cases")
