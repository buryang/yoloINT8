import argparse
import os
import sys
import subprocess
import glob
import json
import traceback
from datetime import datetime

LOG_FILE = "quantize.log"

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

def output_result(success, onnx_path=None, param_path=None, bin_path=None, error=None, error_type=None):
    result = {
        "success": success,
        "onnx_path": onnx_path,
        "param_path": param_path,
        "bin_path": bin_path,
        "error": error,
        "error_type": error_type,
        "log_file": LOG_FILE
    }
    print("\n" + "="*50)
    print("RESULT:" + json.dumps(result))
    print("="*50 + "\n")
    return result

try:
    import onnx
    import numpy
except ImportError:
    print("Error: onnx/numpy not installed. Run: pip install onnx numpy")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed. Run: pip install opencv-python")
    sys.exit(1)


def find_ncnn_tools():
    base_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        ".",
        "..",
        os.path.expanduser("~"),
        "C:/",
        os.path.expanduser("~/ncnn"),
        "/usr/local",
        "/opt",
    ]
    
    patterns = [
        "*ncnn*/x64/bin/cnnoptimize.exe",
        "*ncnn*/x64/bin/cnnoptimize",
        "*ncnn*/bin/cnnoptimize",
        "*ncnn*/bin/onnx2ncnn",
        "cnnoptimize.exe",
        "cnnoptimize",
    ]
    
    for base in base_dirs:
        for pattern in patterns:
            matches = glob.glob(os.path.join(base, pattern), recursive=True)
            if matches:
                return os.path.dirname(matches[0])
    
    import shutil
    if shutil.which('cnnoptimize'):
        return ""
    
    return None


def find_ncnn_tool(tool_name):
    ncnn_dir = find_ncnn_tools()
    if ncnn_dir and os.path.isdir(ncnn_dir):
        exe = os.path.join(ncnn_dir, tool_name)
        if os.path.exists(exe):
            return exe

        exe = os.path.join(ncnn_dir, tool_name + ".exe")
        if os.path.exists(exe):
            return exe

    import shutil
    res = shutil.which(tool_name)
    if res:
        return res
    return None


def letterbox_preprocess(img, target_size, mean, norm, pad_value=114):
    h, w = img.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h))
    padded = numpy.full((target_size, target_size, 3), pad_value, dtype=numpy.uint8)
    padded[:new_h, :new_w] = resized
    
    img_float = padded.astype(numpy.float32)
    for i in range(3):
        img_float[:, :, i] = (img_float[:, :, i] - mean[i]) * norm[i]
    
    return img_float.transpose(2, 0, 1)


class ONNXCalibrator:
    def __init__(self, image_dir, img_size=640, max_samples=100, mean=None, norm=None):
        self.image_dir = image_dir
        self.img_size = img_size
        self.max_samples = max_samples
        self.mean = mean or [0.0, 0.0, 0.0]
        self.norm = norm or [1.0/255, 1.0/255, 1.0/255]
        self.inputs = []
        
    def collect(self):
        if not os.path.isdir(self.image_dir):
            print(f"[WARNING] Calibration directory not found: {self.image_dir}")
            return False
        
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for f in os.listdir(self.image_dir):
            ext = os.path.splitext(f.lower())[1]
            if ext in valid_exts:
                image_files.append(os.path.join(self.image_dir, f))
        
        image_files = image_files[:self.max_samples]
        if not image_files:
            print(f"[WARNING] No valid images in {self.image_dir}")
            return False
        
        print(f"[INFO] Collecting {len(image_files)} calibration images...")
        count = 0
        for path in image_files:
            img = cv2.imread(path)
            if img is None:
                continue
            self.inputs.append(letterbox_preprocess(img, self.img_size, self.mean, self.norm))
            count += 1
        print(f"[INFO] Collected {count} images")
        return count > 0


def verify_onnx_model(model_path):
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        return True
    except:
        return False


def generate_calibration_table(onnx_path, param_path, bin_path, calibration_dir):
    ncnn2table = find_ncnn_tool("ncnn2table")
    if not ncnn2table:
        print("[ERROR] ncnn2table not found")
        return None

    image_list = []
    for f in os.listdir(calibration_dir):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_list.append(os.path.join(calibration_dir, f))
    image_list = image_list[:100]

    list_file = os.path.join(calibration_dir, "images.txt")
    with open(list_file, 'w') as f:
        f.write("\n".join(image_list))

    table_path = os.path.splitext(onnx_path)[0] + ".table"

    cmd = [
        ncnn2table,
        param_path,
        bin_path,
        list_file,
        table_path,
        "0=0,0,0",
        "1=1,1,1",
        "640",
        "640",
        "0",
        "255"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"[INFO] Calibration table saved: {table_path}")
        return table_path
    except:
        return None


def convert_to_ncnn(onnx_path, use_quant=False, calibration_dir=None):
    cnnoptimize = find_ncnn_tool("cnnoptimize")
    if not cnnoptimize:
        return False, None, None, "TOOL_NOT_FOUND", "cnnoptimize not found"

    base = os.path.splitext(onnx_path)[0]
    param = base + ".param"
    bin = base + ".bin"

    try:
        subprocess.run([cnnoptimize, onnx_path, param, bin, "1"], check=True)
    except:
        return False, None, None, "CONVERT_FAILED", "cnnoptimize failed"

    if not use_quant or not calibration_dir or not os.path.isdir(calibration_dir):
        return True, param, bin, None, None

    table = generate_calibration_table(onnx_path, param, bin, calibration_dir)
    if not table:
        return True, param, bin, None, None

    ncnn2int8 = find_ncnn_tool("ncnn2int8")
    if not ncnn2int8:
        return True, param, bin, None, None

    q_param = base + "_int8.param"
    q_bin = base + "_int8.bin"

    try:
        subprocess.run([ncnn2int8, param, bin, q_param, q_bin, table], check=True)
        return True, q_param, q_bin, None, None
    except:
        return True, param, bin, None, None


def main():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--calibration-images")
    parser.add_argument("--to-ncnn", action="store_true")
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify and not verify_onnx_model(args.input):
        output_result(False, onnx_path=args.input, error="ONNX invalid")
        return

    if args.to_ncnn:
        ok, p, b, et, em = convert_to_ncnn(
            args.input,
            use_quant=args.quant,
            calibration_dir=args.calibration_images
        )
        if ok:
            output_result(True, onnx_path=args.input, param_path=p, bin_path=b)
        else:
            output_result(False, onnx_path=args.input, error=em, error_type=et)
    else:
        log("Skipping NCNN conversion (use --to-ncnn to convert)")


if __name__ == "__main__":
    main()