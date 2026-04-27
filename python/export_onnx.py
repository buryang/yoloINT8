import argparse
import sys
import os
import shutil
import json
import traceback
from datetime import datetime
from pathlib import Path

LOG_FILE = "export_onnx.log"

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}"
    print(log_entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

def log_error(e=None):
    if e:
        log(f"ERROR: {str(e)}", "ERROR")
        log(traceback.format_exc(), "ERROR")
    else:
        log("Unknown error occurred", "ERROR")

def output_result(success, onnx_path=None, error=None, error_type=None):
    result = {
        "success": success,
        "onnx_path": onnx_path,
        "error": error,
        "error_type": error_type,
        "log_file": LOG_FILE
    }
    print("\n" + "="*50)
    print("RESULT:" + json.dumps(result))
    print("="*50 + "\n")
    return result

try:
    import torch
except ImportError:
    print("Error: torch not installed. Run: pip install torch")
    sys.exit(1)

try:
    import onnx
    import numpy
except ImportError:
    print("Error: onnx/numpy not installed. Run: pip install onnx numpy")
    sys.exit(1)


def check_requirements():
    missing = []
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    try:
        import onnxruntime
    except ImportError:
        missing.append("onnxruntime")
    
    if missing:
        print(f"Warning: Missing packages: {', '.join(missing)}")


def load_yolov5(weights):
    if not os.path.exists(weights):
        raise FileNotFoundError(f"Weights file not found: {weights}")

    try:
        ckpt = torch.load(weights, map_location='cpu', weights_only=False)
        
        if 'model' in ckpt:
            model = ckpt['model'].float()
        elif 'ema' in ckpt:
            model = ckpt['ema'].float()
        elif isinstance(ckpt, torch.nn.Module):
            model = ckpt.float()
        else:
            raise ValueError("Unknown YOLOv5 checkpoint format")
        
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLOv5 model: {e}")


def export_yolov5_to_onnx(weights, img_size=640, fp16=False):
    log(f"Loading model from {weights}")
    
    try:
        model = load_yolov5(weights)
    except Exception as e:
        log_error(e)
        return None, "LOAD_FAILED", str(e)

    output_path = os.path.splitext(weights)[0] + ".onnx"
    if fp16:
        output_path = os.path.splitext(weights)[0] + "_fp16.onnx"

    log(f"Exporting to {output_path}")

    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    if fp16:
        model = model.half()
        dummy_input = dummy_input.half()

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=12,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes=None,
            verbose=False,
            training=False
        )
        
        if not os.path.exists(output_path):
            log("Export failed, file not created", "ERROR")
            return None, "EXPORT_FAILED", "File not created"
        
        log(f"Export completed: {output_path}")
        return output_path, None, None
        
    except Exception as e:
        log_error(e)
        return None, "EXPORT_FAILED", str(e)


def export_yolov8_to_onnx(weights, img_size=640, fp16=False):
    try:
        from ultralytics import YOLO
    except ImportError:
        log("ultralytics not installed. Run: pip install ultralytics", "ERROR")
        return None, "DEPENDENCY_MISSING", "ultralytics not installed"

    log(f"Loading model from {weights}")
    
    if not os.path.exists(weights):
        log(f"Weights file not found: {weights}", "ERROR")
        return None, "FILE_NOT_FOUND", f"Weights not found: {weights}"

    try:
        model = YOLO(weights)
    except Exception as e:
        log_error(e)
        return None, "LOAD_FAILED", str(e)
    
    output_dir = os.path.dirname(weights) or "."
    output_name = os.path.splitext(os.path.basename(weights))[0] + ".onnx"
    if fp16:
        output_name = os.path.splitext(os.path.basename(weights))[0] + "_fp16.onnx"
    output_path = os.path.join(output_dir, output_name)

    log(f"Exporting to {output_path}")

    try:
        model.export(
            format='onnx',
            imgsz=img_size,
            half=fp16,
            simplify=False,
            opset=12,
            dynamic=False
        )
        
        expected_path = os.path.join(output_dir, os.path.basename(weights).replace('.pt', '.onnx'))
        if fp16:
            expected_path = os.path.join(output_dir, os.path.basename(weights).replace('.pt', '_fp16.onnx'))
        
        if os.path.exists(expected_path) and expected_path != output_path:
            shutil.move(expected_path, output_path)
        
        if not os.path.exists(output_path):
            for f in os.listdir(output_dir):
                if f.endswith('.onnx'):
                    output_path = os.path.join(output_dir, f)
                    break
        
        log(f"Export completed: {output_path}")
        return output_path, None, None
        
    except Exception as e:
        log_error(e)
        return None, "EXPORT_FAILED", str(e)


def detect_model_type(weights):
    if not os.path.exists(weights):
        return None, "FILE_NOT_FOUND", f"Weights not found: {weights}"
    
    basename = os.path.basename(weights).lower()
    
    if 'yolov8' in basename or 'yolo8' in basename:
        return 'v8', None, None
    elif 'yolov5' in basename or 'yolo5' in basename:
        return 'v5', None, None
    elif 'yolov4' in basename or 'yolov3' in basename:
        return 'v5', None, None
    elif basename.endswith('.pt'):
        try:
            ckpt = torch.load(weights, map_location='cpu', weights_only=False)
            if 'model' in ckpt or 'ema' in ckpt:
                return 'v5', None, None
        except:
            pass
    
    log(f"Cannot detect model type from filename: {weights}", "WARNING")
    log("Assuming YOLOv8 (default)")
    return 'v8', None, None


def verify_onnx_model(model_path):
    print(f"[INFO] Verifying ONNX model: {model_path}")
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            dummy = numpy.random.randn(1, 3, 640, 640).astype(numpy.float32)
            sess.run(None, {input_name: dummy})
            print("[INFO] ONNX model verification passed")
            return True
        except ImportError:
            print("[WARNING] onnxruntime not installed, skipping runtime verification")
            return True
        except Exception as e:
            print(f"[WARNING] Runtime verification failed: {e}")
            return False
            
    except Exception as e:
        print(f"[ERROR] ONNX model verification failed: {e}")
        return False


def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    parser = argparse.ArgumentParser(description='Export YOLO to ONNX')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pt weights')
    parser.add_argument('--img-size', type=int, default=640, help='Input image size')
    parser.add_argument('--fp16', action='store_true', help='Export as FP16')
    parser.add_argument('--model-type', type=str, default='auto', 
                        choices=['auto', 'v5', 'v8'], help='Model type')
    parser.add_argument('--verify', action='store_true', help='Verify ONNX model after export')

    args = parser.parse_args()

    if args.img_size % 32 != 0:
        print(f"[WARNING] img_size {args.img_size} not multiple of 32, adjusting")
        args.img_size = ((args.img_size + 31) // 32) * 32

    if args.model_type == 'auto':
        args.model_type, et, em = detect_model_type(args.weights)
        if args.model_type is None:
            log("Cannot determine model type", "ERROR")
            output_result(False, error=em, error_type=et)
            sys.exit(1)

    log(f"Model type: {args.model_type}")

    onnx_path = None
    error_type = None
    error_msg = None
    
    if args.model_type == 'v8':
        onnx_path, error_type, error_msg = export_yolov8_to_onnx(args.weights, args.img_size, args.fp16)
    else:
        onnx_path, error_type, error_msg = export_yolov5_to_onnx(args.weights, args.img_size, args.fp16)

    if onnx_path and os.path.exists(onnx_path):
        log(f"ONNX export to {onnx_path}")
        
        if args.verify:
            if verify_onnx_model(onnx_path):
                log("Model verification passed")
            else:
                log("Model verification failed", "WARNING")
        
        output_result(True, onnx_path=onnx_path)
        
        log(f"Next step: python quantize.py --input {onnx_path} --to-ncnn")
    else:
        log(f"Export failed: {error_msg}", "ERROR")
        output_result(False, error=error_msg, error_type=error_type)
        sys.exit(1)


if __name__ == '__main__':
    check_requirements()
    main()