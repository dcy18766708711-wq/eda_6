# PP-OCRv6 manual fine-tuned model

This folder contains the exported best inference models from the manual OCR
fine-tuning run. It is intended to be copied to another computer as a model
package; the Python environment itself is not included.

## Contents

- `models/det/`: fine-tuned text detection model.
- `models/rec/`: fine-tuned text recognition model and its character dictionary.
- `.paddlex/`: cached text-line orientation model used by the runner, so the
  first inference does not need to download it.
- `run_ppocrv6_medium.py`: inference runner.
- `requirements*.txt`: package versions and CPU/GPU runtime choices.

## Install the runtime

Create a Python environment on the target computer and install PaddleOCR 3.x
with exactly one matching PaddlePaddle CPU or GPU build:

```bash
python -m pip install -r requirements.txt
# CPU computer:
python -m pip install -r requirements-cpu.txt
# NVIDIA GPU computer, with a compatible CUDA runtime:
python -m pip install -r requirements-gpu.txt
```

The package already includes the text-line orientation cache. Keep the
directory structure unchanged so the runner can find `.paddlex/` beside it.

## Run on one image

From this directory. The local detection and recognition models are used by
default, so no absolute Linux path is required:

```bash
python run_ppocrv6_medium.py \
  --image /path/to/input.png \
  --output ./results \
  --device cpu
```

For a CUDA installation, change `--device cpu` to `--device gpu:0`.

The JSON result and rendered OCR image are written below `./results`.

The supplied source environment used PaddleOCR `3.7.0`, PaddleX `3.7.2`,
PaddlePaddle `3.3.0`, NumPy `2.3.5`, SciPy `1.18.1`, and OpenCV
`4.10.0.84`.

## Important

The files in `models/` are exported inference weights, not training
checkpoints. They are suitable for OCR inference. The original training
checkpoints remain in the project output directory and are not included here.
