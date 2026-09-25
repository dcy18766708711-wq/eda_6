@echo off
setlocal

REM Visualize all annotations in 200_train_cases.
REM Run this file from the project root or double-click it.
REM Output: outputs\annotation_preview\200_train_cases

cd /d "%~dp0\.."

python ".\scripts\visualize_annotations.py" ^
  --subset 200_train_cases ^
  --coord-system bottom-left ^
  --output-dir ".\outputs\annotation_preview"

if errorlevel 1 (
  echo.
  echo [ERROR] Visualization failed.
  pause
  exit /b 1
)

echo.
echo [DONE] Annotation previews are in:
echo %cd%\outputs\annotation_preview\200_train_cases
pause
