# cpntLibrary.json 字段说明

## 顶层结构

根对象是一个字典：

```text
{
  "<元件名>": { ... 元件记录 ... },
  ...
}
```

| 部分 | 说明 |
|------|------|
| **键** | 元件在库中的唯一名称，检索主键 |
| **值** | 元件记录，固定含 `esym_path`、`pins`、`same_hash_components` |

### 键名常见形态

| 形态 | 示例 | 含义 |
|------|------|------|
| 基础型号名 | `AMS1117-3.3`、`CH340C` | 通用符号名 |
| 带料号后缀 | `AMS1117-3.3_C6186` | 同一器件的不同封装/料号变体（`_C` + 数字多为嘉立创料号） |
| 其它后缀 | `AMS1117-3.3_1`、`CH340C_JX` | 同名不同符号版本 |
| 描述性名称 | `1.25-2PWB`、`2.54-1x6P直针` | 连接器、排针等 |

---

## 元件记录字段

```json
{
  "esym_path": "AMS1117-3.3.esym",
  "pins": [ ... ],
  "same_hash_components": [ ... ]
}
```

### `esym_path`（string）

对应 EasyEDA 符号文件（`.esym`）的文件名或相对路径，用于定位原始符号图形。通常与库键名相近，特殊字符可能被替换（如 `*` → `_`）。

### `pins`（array of object）

该元件的引脚信息列表。多为多个同源符号实例上引脚标注的**聚合**，因此：

- 同一 `number` 可能对应多种 `name`（如 `VOUT` / `Out` / `Output`）
- 同一 `pin_id` 可能出现多条，`pin_type` 也可能不同
- 条目数可能远大于物理引脚数

#### 引脚对象字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `pin_id` | string | 符号内部引脚图形/节点 ID。常见前缀：`e`（如 `e4`）、`ie`（如 `ie8`）。**不是**封装焊盘号，在数组内**不保证唯一**。 |
| `name` | string | 引脚电气名称/丝印名，如 `GND`、`VIN`、`SCL`。 |
| `number` | string | 引脚编号（pin number），字符串形式，如 `"1"`、`"2"`。按管脚号归类时以该字段为准。 |
| `pin_type` | string | 引脚电气类型，见下表。 |

#### `pin_type` 取值

| 值 | 含义 |
|----|------|
| `IN` | 输入 |
| `OUT` | 输出 |
| `BI` | 双向（Bidirectional） |
| `Passive` | 无源 |
| `Power` | 电源 |
| `Undefined` | 未定义 / 未知 |
| `""`（空字符串） | 缺失或未标注 |

### `same_hash_components`（array of object）

与当前条目符号哈希相同（或视为同源）的其它工程中的元件引用，用于溯源。

| 字段 | 类型 | 说明 |
|------|------|------|
| `project` | string | 来源工程/项目编号（如 `"94"`） |
| `title` | string | 该工程中元件标题/名称，多数与库键相同或接近 |
| `symbol` | string | 符号内容哈希（32 位十六进制），用于判定符号是否相同 |

---

## 示例

```json
{
  "1.25-2PWB": {
    "esym_path": "1.25-2PWB.esym",
    "pins": [
      {
        "pin_id": "e4",
        "name": "1",
        "number": "1",
        "pin_type": "IN"
      },
      {
        "pin_id": "e5",
        "name": "1",
        "number": "1",
        "pin_type": "Undefined"
      }
    ],
    "same_hash_components": [
      {
        "project": "33",
        "title": "1.25-2PWB",
        "symbol": "17f6704f1c59423a890ef3ffb9780ffd"
      }
    ]
  }
}
```
