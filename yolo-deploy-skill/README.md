# YOLO Deploy Skill

OpenCLAW Skill for YOLO model deployment - 自动化完成 YOLO 模型的导出、量化、NCNN 转换和推理测试。

## 功能

- ✅ 模型导出 (PT → ONNX)
- ✅ 模型量化 (FP32/FP16/INT8)
- ✅ NCNN 转换与优化
- ✅ 推理测试
- ✅ 结果可视化

## 架构

```
yolo-deploy-skill/
├── yolo_deploy.py          # 核心工具（调用现有脚本）
├── manifest.yaml           # Skill 定义
├── openclaw.json           # Agent 配置
├── prompts/
│   ├── SOUL.md            # Agent 性格
│   └── workflow.md        # 工作流定义
├── templates/              # 配置模板
└── README.md              # 本文档
```

## 快速开始

### 1. 直接使用 Python 脚本

```bash
# 完整流程
python yolo-deploy-skill/yolo_deploy.py full --weights yolov8n.pt --image test.jpg

# 仅导出 ONNX
python yolo-deploy-skill/yolo_deploy.py export --weights yolov8n.pt

# 仅量化转换
python yolo-deploy-skill/yolo_deploy.py quantize --onnx yolov8n.onnx --to-ncnn

# 仅推理测试
python yolo-deploy-skill/yolo_deploy.py inference --param yolo.param --bin yolo.bin --image test.jpg
```

### 2. 作为 OpenCLAW Skill 使用

将 `yolo-deploy-skill/` 目录复制到 OpenCLAW 的 skills 目录，然后配置 Agent:

```json
{
  "agents": [
    {
      "name": "yolo-deploy",
      "skill": "yolo-deploy",
      "model": "claude-sonnet-4-20250514"
    }
  ]
}
```

## 使用示例

### 完整部署流程

```bash
python yolo_deploy.py full \
  --weights yolov8n.pt \
  --image test.jpg \
  --fp16 \
  --quant \
  --calibration-images ./calibration
```

输出:
```
Step 1: 导出 ONNX
✓ ONNX 导出成功: yolov8n.onnx

Step 2: 量化并转换 NCNN
✓ NCNN 转换成功:
  Param: yolo.param
  Bin: yolo.bin

Step 3: 推理测试
✓ 推理成功: 3 个检测框
  person: 95%
  bicycle: 87%
  car: 78%
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--weights` | PyTorch 权重文件 | 必需 |
| `--onnx` | ONNX 模型文件 | 必需 |
| `--param` | NCNN param 文件 | 必需 |
| `--bin` | NCNN bin 文件 | 必需 |
| `--image` | 测试图像 | 可选 |
| `--output` | 输出图像路径 | result.jpg |
| `--fp16` | FP16 导出 | false |
| `--quant` | INT8 量化 | false |
| `--calibration-images` | 校准图像目录 | 可选 |
| `--conf` | 置信度阈值 | 0.25 |
| `--iou` | NMS IoU 阈值 | 0.45 |
| `--threads` | 线程数 | 4 |

## 与现有脚本的关系

```
yolo-deploy-skill (Skill 包装层)
        ↓ 调用
python/ (核心逻辑 - 直接维护)
  ├── export_onnx.py
  └── quantize.py

cpp/ (推理实现)
  └── main.cpp
```

**优势**: 修改 python/ 下的脚本后，Skill 自动受益，无需额外维护。

## 依赖

- Python 3.8+
- torch
- ultralytics
- onnx
- onnxruntime
- opencv-python

## License

MIT