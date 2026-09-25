@echo off
setlocal

REM Convert 200_train_cases component bbox/type annotations to YOLO format.
REM Output: outputs\yolo_components

cd /d "%~dp0\.."

python ".\scripts\convert_components_to_yolo.py" ^
  --subset 200_train_cases ^
  --class-mode all ^
  --val-ratio 0.2 ^
  --seed 42 ^
  --overwrite

if errorlevel 1 (
  echo.
  echo [ERROR] Conversion failed.
  pause
  exit /b 1
)

echo.
echo [DONE] YOLO dataset is in:
echo %cd%\outputs\yolo_components
pause
