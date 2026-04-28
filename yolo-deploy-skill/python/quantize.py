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
    # Prefer old version (ncnn-20230517) for INT8 quantization
    base_dirs = [
        "D:/ncnn/ncnn-20230517/ncnn-20230517-windows-vs2019/x64/bin",
        "D:/ncnn/ncnn-20260113-windows-vs2022/x64/bin",
        "D:/ncnn/ncnn-20231219-windows-vs2022/x64/bin",
        os.path.dirname(os.path.abspath(__file__)),
        ".",
        "..",
        os.path.expanduser("~"),
        "C:/",
        "D:/",
        os.path.expanduser("~/ncnn"),
        "/usr/local",
        "/opt",
    ]
    
    patterns = [
        "*/x64/bin/ncnnoptimize.exe",
        "*/x64/bin/ncnnoptimize",
        "*/bin/ncnnoptimize.exe",
        "*/bin/ncnnoptimize",
        "ncnnoptimize.exe",
        "ncnnoptimize",
    ]
    
    for base in base_dirs:
        if not os.path.exists(base):
            continue
        for pattern in patterns:
            full_pattern = os.path.join(base, pattern)
            matches = glob.glob(full_pattern, recursive=True)
            if matches:
                return os.path.dirname(matches[0])
    
    return None


def find_ncnn_tool(tool_name):
    # pnnx is the recommended tool for ONNX conversion
    import shutil
    if tool_name == "pnnx" or tool_name == "onnx2ncnn":
        res = shutil.which("pnnx")
        if res:
            return res
        
        # Also check common install locations
        pnnx_paths = [
            os.path.expanduser("~/AppData/Roaming/Python/Python314/Scripts/pnnx.exe"),
            os.path.expanduser("~/AppData/Roaming/Python/Python313/Scripts/pnnx.exe"),
            os.path.expanduser("~/AppData/Roaming/Python/Python312/Scripts/pnnx.exe"),
        ]
        for p in pnnx_paths:
            if os.path.exists(p):
                return p
    
    # For INT8 quantization, prefer old version tools (ncnn-20230517) which work correctly
    # Fall back to newer version if old not found
    ncnn_versions = [
        "D:/ncnn/ncnn-20230517/ncnn-20230517-windows-vs2019/x64/bin",
        "D:/ncnn/ncnn-20260113-windows-vs2022/x64/bin",
        "D:/ncnn/ncnn-20231219-windows-vs2022/x64/bin",
    ]
    
    for base_dir in ncnn_versions:
        tool_path = os.path.join(base_dir, tool_name + ".exe")
        if os.path.exists(tool_path):
            return tool_path
        
        # Also try without .exe
        tool_path_nx = os.path.join(base_dir, tool_name)
        if os.path.exists(tool_path_nx):
            return tool_path_nx
    
    # Try system PATH
    res = shutil.which(tool_name + ".exe") or shutil.which(tool_name)
    if res:
        return res
    
    return None
    
    ncnn_dir = find_ncnn_tools()
    if ncnn_dir and os.path.isdir(ncnn_dir):
        # Try exact name
        exe = os.path.join(ncnn_dir, tool_name)
        if os.path.exists(exe):
            return exe
        
        # Try with .exe
        exe = os.path.join(ncnn_dir, tool_name + ".exe")
        if os.path.exists(exe):
            return exe

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
        "mean=[0.0,0.0,0.0]",
        "norm=[0.00392,0.00392,0.00392]",
        "shape=[640,640,3]",
        "pixel=BGR",
        "method=aciq"
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, errors='ignore')
        if os.path.exists(table_path):
            print(f"[INFO] Calibration table saved: {table_path}")
            return table_path
    except Exception as e:
        print(f"[ERROR] ncnn2table failed: {e}")
    
    return None


def convert_to_ncnn(onnx_path, use_quant=False, calibration_dir=None):
    pnnx = find_ncnn_tool("pnnx")
    if not pnnx:
        return False, None, None, "TOOL_NOT_FOUND", "pnnx not found (run: pip install pnnx)"

    base = os.path.splitext(onnx_path)[0]
    param = base + ".ncnn.param"
    bin = base + ".ncnn.bin"

    try:
        subprocess.run([pnnx, onnx_path], check=True, capture_output=True, text=True)
        # pnnx outputs yolov8n.ncnn.param and yolov8n.ncnn.bin
        expected_param = base + ".ncnn.param"
        expected_bin = base + ".ncnn.bin"
        if os.path.exists(expected_param) and os.path.exists(expected_bin):
            param = expected_param
            bin = expected_bin
        else:
            return False, None, None, "CONVERT_FAILED", "pnnx failed to produce output"
    except subprocess.CalledProcessError as e:
        return False, None, None, "CONVERT_FAILED", f"pnnx failed: {e.stderr}"
    except Exception as e:
        return False, None, None, "CONVERT_FAILED", str(e)

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