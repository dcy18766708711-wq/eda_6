# IC 库索引

按照赛题群最新回复，库的权威顺序为：

1. `BoxICLibrary_200cases.json`：主库，以它为准。
2. `cpntLibrary.json`：只有主库缺失时才作为兼容备用库。

索引只提供 IC 名称候选、pin number、pinname、pin_type、same_hash_components
别名和溯源。索引不提供当前图片中的坐标。

库中同一个 `number` 可能有多个 `name`、`pin_id` 或 `pin_type`，索引会保留全部
候选；物理 pin 数量使用不同 `number` 的数量，不能使用原始 `pins` 数组长度。

构建索引：

```powershell
python index.py `
  --library .\BoxICLibrary_200cases.json `
  --fallback .\cpntLibrary.json `
  --output .\BoxICLibrary_200cases.index.json
```

查询 IC：

```powershell
python index.py --query STM32F103
```

代码调用：

```python
from ic_library.index import load_index

index = load_index("BoxICLibrary_200cases.index.json")
candidates = index.find_candidates("STM32F103")
pins = index.pins(candidates[0]["name"])
```
