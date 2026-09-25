# 电路图识别赛题数据集说明

本数据集用于电路图识别赛题，包含训练数据集 `200_train_cases` 和小批量测试数据集 `10_GTcase`。

## 数据集概览

| 目录 | 样本数 | 用途 | 单个样本内容 |
| --- | ---: | --- | --- |
| `200_train_cases` | 200 | 供选手训练、调试和验证模型 | 1 张电路图原图（PNG）和 1 个赛题格式标注文件（JSON） |
| `10_GTcase` | 10 | 供选手进行小批量测试的 Golden Case | 1 张电路图原图（PNG）和 1 个对应的标准答案文件（JSON） |

## 文件夹结构

```text
Race/
├── README.md
├── 200_train_cases/                 # 训练数据，共 200 个 case
│   ├── 0001/
│   │   ├── 0001-KiCad.png           # 电路图原图
│   │   └── 0001-KiCad_target.json   # 对应的赛题格式标注
│   ├── 0002/
│   │   ├── 0002-Datasheet.png
│   │   └── 0002-Datasheet_target.json
│   ├── ...
│   └── 0200/
│       ├── 0200-jlc.png
│       └── 0200-jlc_target.json
└── 10_GTcase/                       # 小批量 Golden Case，共 10 个 case
    ├── 0025-Altium Designer/
    │   ├── 0025-Altium Designer.png
    │   └── 0025-Altium Designer_target.json
    ├── 0030-Datasheet/
    │   ├── 0030-Datasheet.png
    │   └── 0030-Datasheet_target.json
    ├── ...
    └── 0100-Datasheet/
        ├── 0100-Datasheet.png
        └── 0100-Datasheet_target.json
```

> 目录树仅展示部分样例。实际数据中，`200_train_cases` 含 200 个子文件夹，`10_GTcase` 含 10 个子文件夹。

## 数据内容

### 1. `200_train_cases`

`200_train_cases` 是供选手使用的训练数据集。每个编号子文件夹代表一个独立 case，并包含：

- `*.png`：待识别的电路图原图；
- `*_target.json`：与原图一一对应的赛题格式标注数据。

图片和 JSON 文件使用相同的文件名前缀，可以据此完成配对。例如：

```text
0001-KiCad.png
0001-KiCad_target.json
```

文件名中的 `KiCad`、`Altium Designer`、`Datasheet`、`jlc`、`other` 等文本用于表示样本的来源或类型。

### 2. `10_GTcase`

`10_GTcase` 是选手可直接使用的小批量测试数据集，即 Golden Case。其数据格式与训练集保持一致，可用于：

- 快速检查数据读取和预处理流程；
- 调试模型输出格式；
- 对识别结果进行小规模自测；
- 验证评测脚本或结果解析逻辑。

每个 Golden Case 同样包含一张 PNG 电路图原图及一个对应的标准答案 JSON 文件。

## JSON 标注格式

每个 `*_target.json` 文件的顶层包含以下字段：

```json
{
  "components": {},
  "pins": {},
  "nets": {}
}
```

### `components`

记录电路图中的元器件。每个元器件通常包含：

- `Name`：元器件名称或位号；
- `type`：元器件类型；
- `value`：元器件参数值，未标注时可能为 `null`；
- `bbox`：元器件边界框，格式为 `[x_min, y_min, x_max, y_max]`。

示例：

```json
"C6": {
  "Name": "C6",
  "type": "c",
  "value": "4.7uF",
  "bbox": [220.0, 794.0, 236.0, 814.0]
}
```

### `pins`

按元器件记录引脚信息。每个引脚通常包含：

- `pinname`：引脚名称；
- `point`：引脚在原图中的位置坐标，包含 `x` 和 `y`。

示例：

```json
"U1": {
  "pin_1": {
    "pinname": "GPIO4",
    "point": {
      "x": 540.0,
      "y": 854.0
    }
  }
}
```

### `nets`

记录电路网络及其连线信息。每个网络通常包含：

- `hyperGraph`：该网络关联的元器件及引脚关系；
- `edges`：组成该网络的线段集合，每条线段由若干坐标点表示。

示例：

```json
"net_1": {
  "hyperGraph": "(U1.7)",
  "edges": {
    "edge_1": [
      {"x": 543.0, "y": 784.0},
      {"x": 609.0, "y": 784.0}
    ]
  }
}
```

## 使用说明

1. 遍历对应数据集下的 case 子文件夹；
2. 通过共同的文件名前缀匹配 PNG 原图和 `_target.json` 文件；
3. 使用 `200_train_cases` 进行模型训练与开发；
4. 使用 `10_GTcase` 进行小批量测试和输出格式校验。

坐标均对应各 case 的原始 PNG 图像坐标系。读取数据时请保留浮点数精度，并注意 `value` 等字段可能为 `null`。
