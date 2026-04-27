# yoloINT8
export yolo model to ncnn and do ptq calibration

## YOLOv8N NCNN INT8 量化部署
YOLOv8N 模型转 NCNN + INT8 量化全流程与核心踩坑总结

---

## 核心观点（行业共识）
模型量化难度已大幅降低，自动化工具可完成大部分工作。
实际部署中，**真正决定量化成败的是模型选型**：
结构干净、无 Softmax / DFL 的模型极易量化；
含 DFL、Softmax、小数值回归的模型天生不适合 INT8。

---

## 量化核心坑点
- **DFL 结构对 INT8 不友好**
  DFL 本质是带 Softmax 的分布预测，量化后数值分布坍塌，导致框乱、坐标失效。

- **带 Softmax 的层普遍不适合 INT8**
  输出为概率分布，数值小、差异细微，INT8 无法精确表达，量化必掉点。

- **模型内置 NMS 无法量化**
  NCNN 不支持模型内 NMS 算子，导出时必须关闭 `nms=False`。

- **RGB/BGR 顺序错误**
  YOLO 训练使用 BGR，通道顺序错误会直接导致 INT8 模型完全失效。

- **量化校准与推理预处理不一致**
  mean、norm、shape、pixel 必须完全对齐，否则精度暴跌。

- **输出格式解析错误**
  输出 shape `[1,84,8400]`，84=4(xywh)+80(cls)，解析错误则无检测框。

- **NMS 必须在 C++ 端手动实现**
  漏检、重复框、框被吞，均来自 NMS 实现或阈值问题。

- **无校准集直接量化**
  会导致 INT8 精度下降 10%~30%，模型完全不可用。

---

## 正确导出 ONNX
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=12,
    nms=False
)


## Skill & Agent

openclaw skill install ./yolo-deploy-skill
openclaw skill enable yolo_deploy

openclaw agent create yolo-deploy-agent
