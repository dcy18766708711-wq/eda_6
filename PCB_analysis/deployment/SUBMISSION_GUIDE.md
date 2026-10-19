# 赛题六当前可运行前置流程

该目录是当前可提交的赛题六实现。它只含与赛题解析直接相关的推理、协议和当前拓扑探索代码；不含训练脚本、公开训练/验证数据、实验消融、临时结果或已删除的 `your_program`。

## 已整合的阶段

```text
PNG 原理图
  -> ONNX YOLO：元件候选框与类别
  -> PP-OCRv6：全图文字检测与识别
  -> 统一协议：类型映射、左上/左下坐标转换、结构校验
  -> component_builder：组件 key/type/value/bbox 组装
  -> [可选] IC 名称消歧与真实 pin-tip 定位
  -> [可选，探索中] wire 骨架、terminal/junction、线段几何
  -> [可选，探索中] 由 pin 与实测 wire 线段构建 net/hyperGraph
```

默认运行到 `components`：这是当前最稳定、最适合 CPU 的前置结果。pin、wire、net 都已接入统一入口，但其准确性仍处于迭代阶段，因此必须显式打开。

## 目录职责

```text
deployment/
├─ run_front_pipeline.py        # 统一入口
├─ onnx_yolo/                   # best.onnx、预处理、解码、tile 融合
├─ ppocrv6/                     # 检测/识别导出模型与推理封装
├─ common/                      # 坐标、type、JSON 协议、评分口径配置
├─ configs/                     # 官方 type_list 对应映射
├─ stages/component_builder.py  # 组件组装
├─ ic_library/                  # BoxICLibrary 索引，仅作 IC 语义辅助
└─ topology/                    # IC pin、wire、net 的当前实现
```

## CPU 环境

在仓库根目录运行：

```powershell
python -m pip install -r PCB_analysis/deployment/requirements-cpu.txt
```

## 最小可运行命令

只生成 YOLO+OCR 中间结果、正式组件 JSON 和调试侧车文件：

```powershell
python PCB_analysis/deployment/run_front_pipeline.py `
  --image "<原理图.png>" `
  --output-dir PCB_analysis/deployment/outputs/front_pipeline `
  --yolo-device cpu `
  --ocr-device cpu
```

输出位于：

```text
outputs/front_pipeline/
├─ intermediate/  # 左上角 YOLO/OCR 原始中间结果
├─ components/    # 赛题左下角组件 JSON 与调试信息
└─ run_manifest.json
```

## 可选 pin / wire / net

以下命令会完整运行当前已实现的前置和拓扑探索阶段：

```powershell
python PCB_analysis/deployment/run_front_pipeline.py `
  --image "<原理图.png>" `
  --output-dir PCB_analysis/deployment/outputs/full_current `
  --with-ic-pins `
  --with-wire `
  --with-net `
  --no-ic-local-ocr `
  --no-ic-api
```

`--with-net` 依赖 `--with-ic-pins --with-wire`。net 阶段仅使用图像实测的 pin 点和 wire 线段，`snap-tolerance` 被限制为不超过 5 px；它不使用 IC 库顺序、公开答案或 Qwen 生成坐标。

## API 配置与边界

复制 `api_config.example.py` 为 `api_config.py` 后填入赛事提供的 API 信息。真实配置已被 `.gitignore` 排除，禁止提交密钥。

默认所有 API 开关关闭。启用 `--component-api` 或 `--ic-api` 时，Qwen 只用于 OCR/文字语义消歧，不能修改本地检测框、真实 pin-tip 或 wire/edge 坐标。

## 当前边界

- 组件 JSON 会在写出前进行结构和坐标校验。
- 坐标在内部使用左上角；赛题 JSON 中转换为左下角。
- IC 库只辅助型号和可见 pin identity，不补全图中不可见 pin。
- wire/net 已整合为可运行的当前版本，但尚不是经过充分回归验证的稳定结果；提交评测前需单独做 Component、Pin、Wire、Net 回归。
- `experiments/`、`tmp/`、训练资产和公开数据集不属于该可提交流程。
