# 统一协议层

本目录定义赛题六各阶段之间的稳定接口：

- `coordinate.py`：YOLO/OCR 内部左上角坐标与最终左下角坐标互转。
- `type_mapper.py`：YOLO 类别映射到赛题类型列表。最终官方 `type_list` 更新后，
  应通过映射覆盖，不要在检测器代码中散落类型判断。
- `schema_validator.py`：检查 `components/pins/nets` 的字段、坐标、引用和 edge 格式。
- `scoring_profile.yaml`：保存组件框 20 px、pin/edge 5 px 以及电源网络等评测规则。

协议层不负责识别组件、文字、pin 或 wire；它只负责统一表示和拦截格式错误。
