@echo off
setlocal

REM Train YOLO component detector.
REM Make sure outputs\yolo_components\data.yaml exists first.

cd /d "%~dp0\.."

python ".\scripts\train_yolo_components.py" ^
  --data ".\outputs\yolo_components\data.yaml" ^
  --model "yolo11s.pt" ^
  --epochs 100 ^
  --imgsz 1024 ^
  --batch 8 ^
  --device 0 ^
  --name "components_yolo11s"

if errorlevel 1 (
  echo.
  echo [ERROR] Training failed.
  pause
  exit /b 1
)

echo.
echo [DONE] Training outputs are in:
echo %cd%\outputs\yolo_runs\components_yolo11s
pause
