# EDA 赛题六推理模型

本目录保存可部署的 YOLO ONNX 和 PP-OCRv6 推理资源，以及两者的统一推理入口。
不包含训练数据、训练过程输出或真实 API 密钥。

## 目录

```text
deployment/
├── api_config.py
├── run_yolo_ocr.py
├── common/
│   ├── coordinate.py
│   ├── type_mapper.py
│   ├── schema_validator.py
│   └── scoring_profile.yaml
├── configs/
│   └── type_map.json
├── ic_library/
│   ├── BoxICLibrary_200cases.json
│   ├── BoxICLibrary_200cases.index.json
│   ├── cpntLibrary.json
│   └── index.py
├── onnx_yolo/
│   ├── models/best.onnx
│   ├── classes.json
│   ├── onnx_predict.py
│   ├── onnx_postprocess.py
│   ├── requirements.txt
│   └── README.md
└── ppocrv6/
    ├── models/det/
    ├── models/rec/
    ├── .paddlex/official_models/
    ├── run_ppocrv6_medium.py
    ├── requirements*.txt
    └── README.md
```

YOLO 负责元件检测，PP-OCRv6 负责文字检测与识别。运行统一入口：

首次安装 CPU 环境建议使用部署目录的统一依赖文件：

```bash
python -m pip install -r requirements-cpu.txt
```

```bash
python run_yolo_ocr.py \
  --image /path/to/input.png \
  --output-dir ./outputs/yolo_ocr \
  --yolo-device cpu \
  --ocr-device cpu
```

也可以批量处理一个目录：

```bash
python run_yolo_ocr.py \
  --input-dir /path/to/images \
  --output-dir ./outputs/yolo_ocr
```

集成脚本输出中间 JSON，包含 YOLO 组件框、PP-OCRv6 文字框，以及按空间距离
排序的组件—文字候选关联。中间结果使用左上角图像坐标；后续生成赛题 JSON
时必须统一转换为左下角坐标，并继续完成 pin、wire、net 构建。

组件组装：

```bash
python stages/component_builder.py \\
  --input ./outputs/yolo_ocr/example_yolo_ocr.json \\
  --output ./outputs/components/example_components.json \\
  --debug-output ./outputs/components/example_component_debug.json
```

该阶段输出正式的 `components`，并暂时输出空的 `pins`、`nets`；OCR 候选和
角色信息保存在可选 debug sidecar 中，供后续 IC/pin/net 阶段使用。

Qwen API 配置位于 `api_config.py`，当前默认关闭。后续只接入赛题官方 API，
并且只让 API 修正文字内容，不采用 API 返回的坐标。
