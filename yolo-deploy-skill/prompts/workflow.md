# YOLO-Deploy 工作流定义

## 完整流程 (full_pipeline)

### 入口条件
- 用户提供 `.pt` 权重文件路径

### 步骤

```
1️⃣ 检查环境
   ├── 检查 python/ 脚本是否存在
   └── 检查 C++ 编译产物（推理需要）

2️⃣ 导出 ONNX
   ├── 调用 export_onnx.py
   ├── 参数: --weights <path> --img-size 640
   ├── 可选: --fp16 (FP16 导出)
   └── 输出: *.onnx

3️⃣ 量化转换 (可选)
   ├── 调用 quantize.py
   ├── 参数: --input <onnx> --to-ncnn --quant
   ├── 可选: --calibration-images <dir> (需要 INT8 量化)
   └── 输出: *.param, *.bin

4️⃣ 推理测试 (可选)
   ├── 运行 ./build/yolo_inference
   ├── 参数: <param> <bin> <image> -o result.jpg
   └── 输出: 检测结果 + 可视化图像

5️⃣ 汇总结果
   ├── 输出文件列表
   └── 推理性能统计
```

### 退出条件
- 任何步骤失败则终止流程并报告错误
- 成功后显示最终输出文件路径

---

## 部分流程

### 仅导出 (export_only)

```
1️⃣ 导出 ONNX
2️⃣ 验证模型（可选）
3️⃣ 返回 ONNX 路径
```

### 仅量化 (quantize_only)

```
1️⃣ 量化 ONNX
2️⃣ 转换 NCNN
3️⃣ 返回 param/bin 路径
```

### 仅推理 (infer_only)

```
1️⃣ 检查模型文件
2️⃣ 运行推理
3️⃣ 返回检测结果
```

---

## 参数参考

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--weights` | PyTorch 权重文件 | 必需 |
| `--image` | 测试图像 | 可选 |
| `--fp16` | FP16 导出 | false |
| `--quant` | INT8 量化 | false |
| `--calibration-images` | 校准图像目录 | 可选 |
| `--conf` | 置信度阈值 | 0.25 |
| `--iou` | NMS IoU 阈值 | 0.45 |
| `--threads` | 线程数 | 4 |

---

## 输出示例

```
✅ YOLO 部署完成!

📦 输入: yolov8n.pt
📄 输出文件:
   - yolov8n.onnx (导出)
   - yolo.param (NCNN)
   - yolo.bin (NCNN)
   - result.jpg (检测结果)

🧪 推理结果:
   - person: 95%
   - bicycle: 87%
   - car: 78%

⏱️ 推理耗时: 15.2 ms
```