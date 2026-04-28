import cv2
import numpy as np
import ncnn
import sys
import time

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

def letterbox(img, size=640):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    # Convert BGR to RGB
    padded = padded[:, :, ::-1]
    return padded

def load_model(param_path, bin_path):
    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)
    return net

def detect(net, img, conf_thresh=0.25):
    h_img, w_img = img.shape[:2]
    
    # Simple resize to 640x640
    padded = cv2.resize(img, (640, 640))
    
    # Convert BGR to RGB
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    
    # Create float32 mat and normalize to 0-1 by dividing by 255
    # NCNN from_pixels keeps values as-is, so we pre-normalize
    padded = padded.astype(np.float32) / 255.0
    
    # Reshape to CHW format for NCNN
    padded = padded.transpose(2, 0, 1).flatten()
    
    # Create mat from float array
    mat = ncnn.Mat.from_pixels_float(
        padded.tobytes(), ncnn.Mat.PIXEL_RGB, 640, 640
    )
    
    ex = net.create_extractor()
    ex.input("in0", mat)
    
    mat_out = ncnn.Mat()
    ex.extract("out0", mat_out)
    
    results = []
    
    # NCNN stores as [c=1, h=84, w=8400] - flatten gives [84 * 8400]
    out_data = np.array(mat_out).flatten()
    
    # Layout is [ch0_anchor0-8399, ch1_anchor0-8399, ...]
    # Row 0 = cx for all anchors, Row 1 = cy, Row 2 = w, Row 3 = h, Row 4-83 = class probs
    num_anchors = 8400
    
    # For simple resize (640x640), no padding needed
    scale_x = 640 / w_img
    scale_y = 640 / h_img
    
    # Get bbox coordinates for all anchors
    cx_all = out_data[0 * num_anchors:1 * num_anchors]
    cy_all = out_data[1 * num_anchors:2 * num_anchors]
    w_all = out_data[2 * num_anchors:3 * num_anchors]
    h_all = out_data[3 * num_anchors:4 * num_anchors]
    
    # Get class probabilities and apply sigmoid
    class_probs = out_data[4 * num_anchors:84 * num_anchors].reshape(80, num_anchors)
    class_probs = 1 / (1 + np.exp(-class_probs))
    
    # Find max class for each anchor
    max_class = np.argmax(class_probs, axis=0)
    max_conf = np.max(class_probs, axis=0)
    
    print(f"Max conf stats: min={max_conf.min():.3f}, max={max_conf.max():.3f}, mean={max_conf.mean():.3f}")
    
    candidates = []
    for i in range(num_anchors):
        if max_conf[i] > conf_thresh:
            cx = cx_all[i]
            cy = cy_all[i]
            w = w_all[i]
            h = h_all[i]
            
            # Convert from 640x640 to original coords (simple resize, no padding)
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            
            # Scale to original image
            x1 = x1 / scale_x
            y1 = y1 / scale_y
            x2 = x2 / scale_x
            y2 = y2 / scale_y
            
            # Filter valid boxes
            if x1 >= 0 and y1 >= 0 and x2 <= w_img and y2 <= h_img:
                candidates.append({
                    'class': int(max_class[i]),
                    'conf': float(max_conf[i]),
                    'bbox': [x1, y1, x2, y2]
                })
    
    print(f"Found {len(candidates)} candidates after coordinate filter")
    return candidates


def nms(dets, iou_thresh=0.45):
    dets = sorted(dets, key=lambda x: x['conf'], reverse=True)
    result = []
    
    for i, det in enumerate(dets):
        keep = True
        for r in result:
            if det['class'] != r['class']:
                continue
            x1 = max(det['bbox'][0], r['bbox'][0])
            y1 = max(det['bbox'][1], r['bbox'][1])
            x2 = min(det['bbox'][2], r['bbox'][2])
            y2 = min(det['bbox'][3], r['bbox'][3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area1 = (det['bbox'][2]-det['bbox'][0]) * (det['bbox'][3]-det['bbox'][1])
            area2 = (r['bbox'][2]-r['bbox'][0]) * (r['bbox'][3]-r['bbox'][1])
            iou = inter / (area1 + area2 - inter + 1e-6)
            if iou > iou_thresh:
                keep = False
                break
        if keep:
            result.append(det)
    
    return result

def main():
    if len(sys.argv) < 4:
        print("Usage: python test_ncnn.py <param> <bin> <image> [conf] [iou]")
        sys.exit(1)
    
    param_path = sys.argv[1]
    bin_path = sys.argv[2]
    image_path = sys.argv[3]
    conf = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25
    iou_thresh = float(sys.argv[5]) if len(sys.argv) > 5 else 0.45
    
    print(f"Loading model: {param_path}, {bin_path}")
    net = load_model(param_path, bin_path)
    
    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    start = time.time()
    results = detect(net, img, conf)
    results = nms(results, iou_thresh)
    end = time.time()
    
    print(f"\n=== Detection Results ===")
    print(f"Inference time: {(end-start)*1000:.1f} ms")
    print(f"Detections: {len(results)}\n")
    
    for det in results:
        print(f"{CLASS_NAMES[det['class']]}: {det['conf']*100:.1f}% bbox=[{int(det['bbox'][0])},{int(det['bbox'][1])},{int(det['bbox'][2])},{int(det['bbox'][3])}]")
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{CLASS_NAMES[det['class']]} {det['conf']*100:.0f}%"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imwrite("inference/result.jpg", img)
    print(f"\nResult saved to inference/result.jpg")

if __name__ == "__main__":
    main()