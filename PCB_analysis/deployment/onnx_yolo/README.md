# Standalone YOLO ONNX inference

This folder is independent of the custom YOLOv13 Python source. It contains
the class mapping, ONNX preprocessing, output decoding and class-aware NMS.

Install the runtime:

```bash
python -m pip install -r requirements.txt
```

The package contains the selected model at `models/best.onnx`. Run one image:

```bash
python onnx_predict.py \
  --model ./models/best.onnx \
  --image /path/to/input.png \
  --output-dir ./results \
  --device cpu
```

The exported model uses a fixed input size of `1280x1280`; keep the default
`--imgsz 1280`. The ONNX output is decoded and passed through class-aware NMS
by `onnx_postprocess.py`.

The model is independent of the custom YOLOv13 Python source. The original
`.pt` training checkpoint and training outputs are kept outside this deployment
package.

For `onnxruntime-gpu`, use `--device cuda` if `CUDAExecutionProvider` is
available. The output uses top-left image coordinates and the format:

```json
{
  "class_name": "box",
  "confidence": 0.99,
  "bbox_xyxy_top_left": [1.0, 2.0, 3.0, 4.0]
}
```
