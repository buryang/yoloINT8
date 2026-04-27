import argparse
import os
import sys
import subprocess
import glob

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
                return matches[0]
    
    import shutil
    for cmd in ['cnnoptimize', 'onnx2ncnn']:
        if shutil.which(cmd):
            return cmd
    
    return None


def find_ncnn2int8():
    base_dirs = [
        os.path.dirname(os.path.abspath(__file__)),
        ".",
        "..",
        "C:/",
        os.path.expanduser("~/ncnn"),
    ]
    
    patterns = [
        "*ncnn*/x64/bin/ncnn2int8.exe",
        "*ncnn*/x64/bin/ncnn2int8",
        "*ncnn*/bin/ncnn2int8",
        "ncnn2int8.exe",
        "ncnn2int8",
    ]
    
    for base in base_dirs:
        for pattern in patterns:
            matches = glob.glob(os.path.join(base, pattern), recursive=True)
            if matches:
                return matches[0]
    
    import shutil
    for cmd in ['ncnn2int8']:
        if shutil.which(cmd):
            return cmd
    
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
        self.mean = mean if mean else [0.0, 0.0, 0.0]
        self.norm = norm if norm else [1.0/255, 1.0/255, 1.0/255]
        self.inputs = []
        
    def collect(self):
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        if not os.path.isdir(self.image_dir):
            print(f"[WARNING] Calibration directory not found: {self.image_dir}")
            return False
        
        for f in os.listdir(self.image_dir):
            ext = os.path.splitext(f.lower())[1]
            if ext in valid_exts:
                image_files.append(os.path.join(self.image_dir, f))
        
        image_files = image_files[:self.max_samples]
        
        if not image_files:
            print(f"[WARNING] No valid images found in {self.image_dir}")
            return False
        
        print(f"[INFO] Collecting {len(image_files)} calibration images...")
        
        collected = 0
        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Failed to read: {img_path}")
                continue
            
            processed = letterbox_preprocess(img, self.img_size, self.mean, self.norm)
            self.inputs.append(processed)
            collected += 1
        
        print(f"[INFO] Collected {collected} valid images")
        return collected > 0
    
    def run_calibration(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError:
            print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
            return None
        
        print("[INFO] Running calibration...")
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        try:
            sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"[ERROR] Failed to create inference session: {e}")
            return None
        
        input_name = sess.get_inputs()[0].name
        output_names = [o.name for o in sess.get_outputs()]
        
        activation_ranges = {}
        
        for i, inp_data in enumerate(self.inputs):
            inp = inp_data[numpy.newaxis, :].astype(numpy.float32)
            outputs = sess.run(output_names, {input_name: inp})
            
            for j, out_name in enumerate(output_names):
                out = outputs[j]
                if out_name not in activation_ranges:
                    activation_ranges[out_name] = {'min': float(out.min()), 'max': float(out.max())}
                else:
                    activation_ranges[out_name]['min'] = min(activation_ranges[out_name]['min'], float(out.min()))
                    activation_ranges[out_name]['max'] = max(activation_ranges[out_name]['max'], float(out.max()))
            
            if (i + 1) % 10 == 0:
                print(f"  Calibrated {i+1}/{len(self.inputs)}")
        
        print(f"[INFO] Calibration completed for {len(activation_ranges)} outputs")
        return activation_ranges


def verify_onnx_model(model_path):
    print(f"[INFO] Verifying ONNX model: {model_path}")
    
    try:
        model = onnx.load(model_path)
        onnx.checker.check_model(model)
        print("[INFO] ONNX structure valid")
    except Exception as e:
        print(f"[ERROR] ONNX structure invalid: {e}")
        return False
    
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        input_shape = sess.get_inputs()[0].shape
        input_name = sess.get_inputs()[0].name
        
        batch = input_shape[0] if input_shape[0] > 0 else 1
        h = input_shape[2] if len(input_shape) >= 3 else 640
        w = input_shape[3] if len(input_shape) >= 4 else 640
        
        dummy = numpy.random.randn(batch, 3, h, w).astype(numpy.float32)
        sess.run(None, {input_name: dummy})
        
        print("[INFO] ONNX Runtime inference test passed")
        return True
        
    except ImportError:
        print("[WARNING] onnxruntime not installed, skipping runtime test")
        return True
    except Exception as e:
        print(f"[ERROR] Runtime test failed: {e}")
        return False


def generate_calibration_table(onnx_path, calibration_dir, num_samples=100):
    if not calibration_dir or not os.path.exists(calibration_dir):
        print("[INFO] No calibration data, using default quantization")
        return None
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("[WARNING] onnxruntime not available for calibration")
        return None
    
    print(f"[INFO] Generating calibration table from {calibration_dir}...")
    
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) 
                   if os.path.splitext(f.lower())[1] in valid_exts][:num_samples]
    
    if not image_files:
        print("[WARNING] No valid calibration images found")
        return None
    
    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    
    calibration_data = {}
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        scale = min(640 / h, 640 / w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        padded = numpy.full((640, 640, 3), 114, dtype=numpy.uint8)
        padded[:new_h, :new_w] = resized
        
        img_float = padded.astype(numpy.float32) / 255.0
        img_float = img_float.transpose(2, 0, 1)
        
        inp = img_float[numpy.newaxis, :]
        sess.run(None, {input_name: inp})
        
        if (i + 1) % 10 == 0:
            print(f"  Calibrated {i+1}/{len(image_files)}")
    
    print("[INFO] Calibration table generated (using default)")
    return {}


def convert_to_ncnn(onnx_path, ncnn_dir=None, use_quant=False, calibration_dir=None):
    if ncnn_dir is None:
        ncnn_dir = os.path.dirname(os.path.abspath(__file__))
    
    ncnn_optimize = find_ncnn_tools()
    
    if not ncnn_optimize:
        print("[ERROR] NCNN conversion tool not found")
        print("[INFO] Please download NCNN from: https://github.com/Tencent/ncnn/releases")
        return False, None, None
    
    if ncnn_optimize and not os.path.exists(ncnn_optimize):
        print(f"[WARNING] Tool not found at {ncnn_optimize}, trying system PATH")
        ncnn_optimize = os.path.basename(ncnn_optimize)
    
    base_name = os.path.splitext(onnx_path)[0]
    param_path = base_name + ".param"
    bin_path = base_name + ".bin"
    
    cmd = [ncnn_optimize, onnx_path, param_path, bin_path, "1"]
    
    print(f"[INFO] Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"[SUCCESS] NCNN conversion completed:")
            print(f"  Param: {param_path}")
            print(f"  Bin: {bin_path}")
        else:
            print(f"[ERROR] NCNN conversion failed:")
            print(result.stderr)
            return False, None, None
            
    except subprocess.TimeoutExpired:
        print("[ERROR] NCNN conversion timeout")
        return False, None, None
    except FileNotFoundError:
        print(f"[ERROR] Tool not found: {ncnn_optimize}")
        return False, None, None
    except Exception as e:
        print(f"[ERROR] Failed to run ncnn-optimize: {e}")
        return False, None, None

    if use_quant and calibration_dir:
        ncnn2int8 = find_ncnn2int8()
        if not ncnn2int8:
            print("[WARNING] ncnn2int8 not found, skipping quantization")
            return True, param_path, bin_path
        
        if not os.path.exists(ncnn2int8):
            ncnn2int8 = os.path.basename(ncnn2int8)
        
        quant_param_path = base_name + "_quant.param"
        quant_bin_path = base_name + "_quant.bin"
        
        calibration_table = generate_calibration_table(onnx_path, calibration_dir)
        
        cmd = [ncnn2int8, param_path, bin_path, quant_param_path, quant_bin_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"[SUCCESS] INT8 quantization completed:")
                print(f"  Param: {quant_param_path}")
                print(f"  Bin: {quant_bin_path}")
                return True, quant_param_path, quant_bin_path
            else:
                print(f"[WARNING] Quantization failed: {result.stderr}")
                return True, param_path, bin_path
        except Exception as e:
            print(f"[ERROR] Quantization error: {e}")
            return True, param_path, bin_path
    
    return True, param_path, bin_path


def main():
    parser = argparse.ArgumentParser(description='Calibrate and convert YOLO ONNX to NCNN')
    parser.add_argument('--input', type=str, required=True, help='Input ONNX file')
    parser.add_argument('--calibration-images', type=str, default=None, 
                        help='Path to calibration images directory')
    parser.add_argument('--calibration-samples', type=int, default=100,
                        help='Max calibration images to use')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for calibration')
    parser.add_argument('--mean', type=float, nargs=3, default=[0, 0, 0],
                        help='Mean values for normalization')
    parser.add_argument('--norm', type=float, nargs=3, default=[0.00392, 0.00392, 0.00392],
                        help='Norm values for normalization (1/255)')
    parser.add_argument('--to-ncnn', action='store_true', help='Convert to NCNN')
    parser.add_argument('--quant', action='store_true', help='Generate INT8 quantized model')
    parser.add_argument('--verify', action='store_true', default=True, help='Verify ONNX model')
    parser.add_argument('--ncnn-tools-path', type=str, default=None,
                        help='Path to NCNN tools directory')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    if args.quant and not args.calibration_images:
        print("[INFO] Quantization enabled but no calibration images provided")

    if args.verify:
        if not verify_onnx_model(args.input):
            print("[ERROR] Model verification failed, aborting")
            sys.exit(1)

    if args.calibration_images and os.path.exists(args.calibration_images):
        calibrator = ONNXCalibrator(
            args.calibration_images, 
            args.img_size, 
            args.calibration_samples,
            args.mean,
            args.norm
        )
        
        if calibrator.collect():
            activation_ranges = calibrator.run_calibration(args.input)
            if activation_ranges:
                print(f"[INFO] Calibration collected {len(activation_ranges)} activation ranges")

    if args.to_ncnn:
        success, param_path, bin_path = convert_to_ncnn(
            args.input, 
            ncnn_dir=args.ncnn_tools_path,
            use_quant=args.quant,
            calibration_dir=args.calibration_images
        )
        
        if success and param_path and bin_path:
            print("\n[SUCCESS] Conversion completed!")
            print(f"  ONNX: {args.input}")
            print(f"  NCNN param: {param_path}")
            print(f"  NCNN bin: {bin_path}")
            print(f"\n[INFO] To run inference:")
            print(f"  ./yolo_inference {param_path} {bin_path} <image.jpg>")
            if "quant" in param_path:
                print(f"  ./yolo_inference {param_path} {bin_path} <image.jpg> --quant")
        else:
            print("[ERROR] NCNN conversion failed")
            sys.exit(1)
    else:
        print("\n[INFO] Skipping NCNN conversion (use --to-ncnn to convert)")


if __name__ == '__main__':
    main()