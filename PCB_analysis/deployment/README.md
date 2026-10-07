# EDA 赛题六推理模型

本目录只保存可部署的 YOLO ONNX 和 PP-OCRv6 推理资源，不包含训练数据、
训练过程输出或 API 密钥。

## 目录

```text
deployment/
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

YOLO 负责元件检测，PP-OCRv6 负责文字检测与识别。后续赛题主流程应在
上层调用这两个模块，再完成 pin、wire、net 和最终 JSON 构建。
