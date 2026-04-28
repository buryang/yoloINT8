# YOLO Inference Test

## 编译

需要先安装 OpenCV 和 NCNN，然后：

```bash
cd inference
mkdir build && cd build
cmake ..
cmake --build .
```

## 运行

```bash
# FP32 推理
./yolo_test yolov8n.ncnn.param yolov8n.ncnn.bin test.jpg

# INT8 推理
./yolo_test yolov8n_int8.param yolov8n_int8.bin test.jpg

# 自定义阈值
./yolo_test yolov8n.ncnn.param yolov8n.ncnn.bin test.jpg 0.5 0.5
```

## 参数

- param: NCNN param 文件
- bin: NCNN bin 文件
- image: 测试图像
- conf: 置信度阈值 (默认 0.25)
- iou: NMS IoU 阈值 (默认 0.45)

## 输出

- 检测结果打印到终端
- 结果图像保存为 result.jpg