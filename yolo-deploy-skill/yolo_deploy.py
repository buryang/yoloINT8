"""
YOLO Deploy - OpenCLAW Skill
YOLO 模型导出、量化、NCNN 推理全流程自动化

直接调用 python/ 下的脚本实现功能
"""

import subprocess
import os
import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class YOLODeploy:
    """YOLO 部署自动化工具"""
    
    def __init__(self, base_dir: str = None):
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            self.base_dir = Path(__file__).parent.resolve()
        
        self.python_dir = self.base_dir / "python"
        self.cpp_dir = self.base_dir / "cpp"
        self.tools_dir = self.base_dir / "tools" / "ncnn"
        
        self.last_output = ""
        self.last_error = ""
        
    def check_environment(self) -> Dict:
        """检查环境是否就绪"""
        issues = []
        ready = {}
        
        for script in ["export_onnx.py", "quantize.py"]:
            path = self.python_dir / script
            if path.exists():
                ready[script] = str(path)
            else:
                issues.append(f"Missing: {script}")
        
        build_dirs = [
            self.cpp_dir / "build",
            self.cpp_dir / "Release", 
            self.cpp_dir / "cmake-build-release"
        ]
        build_found = any(d.exists() for d in build_dirs)
        
        if build_found:
            ready["cpp_build"] = "found"
        else:
            issues.append("C++ build not found (inference will be skipped)")
        
        return {
            "ready": ready,
            "issues": issues,
            "base_dir": str(self.base_dir)
        }
    
    def _detect_exe_path(self) -> Optional[Path]:
        """检测推理程序路径"""
        candidates = []
        
        if sys.platform == "win32":
            candidates = [
                self.cpp_dir / "build" / "Release" / "yolo_inference.exe",
                self.cpp_dir / "build" / "yolo_inference.exe",
                self.cpp_dir / "cmake-build-release" / "Release" / "yolo_inference.exe",
            ]
        else:
            candidates = [
                self.cpp_dir / "build" / "yolo_inference",
                self.cpp_dir / "cmake-build-release" / "yolo_inference",
            ]
        
        for path in candidates:
            if path.exists():
                return path
        
        return None
        
    def _run_command(self, cmd: str, timeout: int = 300) -> Tuple[int, str, str]:
        """执行命令并返回结果"""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.base_dir)
            )
            self.last_output = result.stdout
            self.last_error = result.stderr
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timeout after {timeout}s"
        except Exception as e:
            return -1, "", str(e)
    
    def _parse_onnx_path(self, output: str) -> Optional[str]:
        """从输出中解析 ONNX 文件路径"""
        result_match = re.search(r'"onnx_path":\s*"([^"]+)"', output)
        if result_match:
            return result_match.group(1)
        
        patterns = [
            r"ONNX export to (.+\.onnx)",
            r"Export completed: (.+\.onnx)",
            r"SUCCESS: ONNX export to (.+\.onnx)"
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()
        return None
    
    def _parse_ncnn_paths(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """从输出中解析 NCNN 文件路径"""
        result_match = re.search(r'"param_path":\s*"([^"]+)"', output)
        bin_match = re.search(r'"bin_path":\s*"([^"]+)"', output)
        if result_match and bin_match:
            return result_match.group(1), bin_match.group(1)
        
        param_pattern = r"NCNN param: (.+\.param)"
        bin_pattern = r"NCNN bin: (.+\.bin)"
        
        param_path = re.search(param_pattern, output)
        bin_path = re.search(bin_pattern, output)
        
        return (
            param_path.group(1).strip() if param_path else None,
            bin_path.group(1).strip() if bin_path else None
        )
    
    def _parse_detections(self, output: str) -> List[Dict]:
        """解析检测结果"""
        detections = []
        pattern = r"\[([^\]]+)\] (\d+\.\d+)% bbox=\[([^,]+), ([^,]+), ([^,]+), ([^\]]+)\]"
        
        for match in re.finditer(pattern, output):
            detections.append({
                "class": match.group(1),
                "confidence": float(match.group(2)),
                "bbox": [
                    float(match.group(3)),
                    float(match.group(4)),
                    float(match.group(5)),
                    float(match.group(6))
                ]
            })
        
        return detections
    
    def _parse_error_info(self, output: str) -> Optional[Dict]:
        """从脚本输出解析错误信息"""
        result_match = re.search(r'RESULT:(\{[^}]+\})', output)
        if result_match:
            try:
                import json
                return json.loads(result_match.group(1))
            except:
                pass
        return None
    
    # ==================== 导出功能 ====================
    
    def export_onnx(
        self,
        weights: str,
        img_size: int = 640,
        fp16: bool = False,
        model_type: str = "auto",
        verify: bool = True
    ) -> Dict:
        """
        导出 ONNX 模型
        
        Args:
            weights: 权重文件路径
            img_size: 输入图像尺寸
            fp16: 是否导出 FP16
            model_type: 模型类型 (auto/v5/v8)
            verify: 是否验证模型
        
        Returns:
            {
                "success": bool,
                "onnx_path": str,
                "error": str
            }
        """
        if not os.path.exists(weights):
            return {"success": False, "onnx_path": None, "error": f"Weights not found: {weights}"}
        
        cmd = f"python {self.python_dir}/export_onnx.py --weights {weights} --img-size {img_size} --model-type {model_type}"
        
        if fp16:
            cmd += " --fp16"
        if verify:
            cmd += " --verify"
        
        print(f"[YOLO-Deploy] Running: {cmd}")
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        if returncode == 0:
            onnx_path = self._parse_onnx_path(stdout)
            if onnx_path:
                return {"success": True, "onnx_path": onnx_path, "error": None}
            
        return {
            "success": False,
            "onnx_path": None,
            "error": stderr or "Export failed"
        }
    
    # ==================== 量化功能 ====================
    
    def quantize(
        self,
        onnx_path: str,
        to_ncnn: bool = True,
        quant: bool = False,
        calibration_images: str = None,
        ncnn_tools_path: str = None
    ) -> Dict:
        """
        量化模型并转换 NCNN
        
        Args:
            onnx_path: ONNX 模型路径
            to_ncnn: 是否转换为 NCNN
            quant: 是否生成 INT8 量化模型
            calibration_images: 校准图像目录
            ncnn_tools_path: NCNN 工具路径
        
        Returns:
            {
                "success": bool,
                "onnx_path": str,
                "param_path": str,
                "bin_path": str,
                "error": str
            }
        """
        if not os.path.exists(onnx_path):
            return {"success": False, "onnx_path": None, "param_path": None, "bin_path": None, "error": f"ONNX not found: {onnx_path}"}
        
        cmd = f"python {self.python_dir}/quantize.py --input {onnx_path}"
        
        if to_ncnn:
            cmd += " --to-ncnn"
        if quant:
            cmd += " --quant"
        if calibration_images:
            cmd += f" --calibration-images {calibration_images}"
        if ncnn_tools_path:
            cmd += f" --ncnn-tools-path {ncnn_tools_path}"
        
        print(f"[YOLO-Deploy] Running: {cmd}")
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        if returncode == 0 and to_ncnn:
            param_path, bin_path = self._parse_ncnn_paths(stdout)
            return {
                "success": True,
                "onnx_path": onnx_path,
                "param_path": param_path,
                "bin_path": bin_path,
                "error": None
            }
        
        return {
            "success": False,
            "onnx_path": onnx_path,
            "param_path": None,
            "bin_path": None,
            "error": stderr or "Quantization failed"
        }
    
    # ==================== 推理功能 ====================
    
    def inference(
        self,
        param_path: str,
        bin_path: str,
        image_path: str,
        output_path: str = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_quant: bool = False,
        threads: int = 4,
        verbose: bool = False
    ) -> Dict:
        """
        运行推理测试
        
        Args:
            param_path: NCNN param 文件
            bin_path: NCNN bin 文件
            image_path: 测试图像
            output_path: 输出图像路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值
            use_quant: 是否使用量化模型
            threads: 线程数
            verbose: 是否详细输出
        
        Returns:
            {
                "success": bool,
                "detections": List[Dict],
                "output_image": str,
                "inference_time": float,
                "error": str
            }
        """
        if not os.path.exists(param_path) or not os.path.exists(bin_path):
            return {"success": False, "detections": [], "output_image": None, "inference_time": 0, "error": "Model files not found"}
        
        if not os.path.exists(image_path):
            return {"success": False, "detections": [], "output_image": None, "inference_time": 0, "error": f"Image not found: {image_path}"}
        
        if output_path is None:
            output_path = str(self.base_dir / "result.jpg")
        
        exe_path = self._detect_exe_path()
        
        if not exe_path:
            return {
                "success": False,
                "detections": [],
                "output_image": None,
                "inference_time": 0,
                "error": "C++ inference binary not found. Please compile the project first."
            }
        
        cmd = f'"{exe_path}" {param_path} {bin_path} {image_path} -o "{output_path}" -conf {conf_threshold} -iou {iou_threshold} -t {threads}'
        
        if use_quant:
            cmd += " --quant"
        if verbose:
            cmd += " -v"
        
        print(f"[YOLO-Deploy] Running: {cmd}")
        
        returncode, stdout, stderr = self._run_command(cmd)
        
        time_match = re.search(r"Inference time: ([\d.]+) ms", stdout)
        inference_time = float(time_match.group(1)) if time_match else 0
        
        detections = self._parse_detections(stdout)
        
        return {
            "success": returncode == 0,
            "detections": detections,
            "output_image": output_path if os.path.exists(output_path) else None,
            "inference_time": inference_time,
            "error": None if returncode == 0 else (stderr or "Inference failed")
        }
    
    # ==================== 完整流程 ====================
    
    def full_pipeline(
        self,
        weights: str,
        test_image: str = None,
        fp16: bool = False,
        quant: bool = False,
        calibration_images: str = None,
        ncnn_tools_path: str = None,
        **inference_kwargs
    ) -> Dict:
        """
        执行完整流程：导出 → 量化 → 转换 → 推理
        
        Args:
            weights: 权重文件路径
            test_image: 测试图像路径
            fp16: 是否 FP16 导出
            quant: 是否量化
            calibration_images: 校准图像目录
            ncnn_tools_path: NCNN 工具路径
            **inference_kwargs: 推理参数
        
        Returns:
            {
                "success": bool,
                "steps": {...},
                "error": str
            }
        """
        steps = {}
        
        print("\n" + "="*50)
        print("Step 1: 导出 ONNX")
        print("="*50)
        
        export_result = self.export_onnx(weights, fp16=fp16)
        steps["export"] = export_result
        
        if not export_result["success"]:
            return {"success": False, "steps": steps, "error": f"Export failed: {export_result['error']}"}
        
        onnx_path = export_result["onnx_path"]
        print(f"✓ ONNX 导出成功: {onnx_path}\n")
        
        print("="*50)
        print("Step 2: 量化并转换 NCNN")
        print("="*50)
        
        quant_result = self.quantize(
            onnx_path,
            to_ncnn=True,
            quant=quant,
            calibration_images=calibration_images,
            ncnn_tools_path=ncnn_tools_path
        )
        steps["quantize"] = quant_result
        
        if not quant_result["success"]:
            return {"success": False, "steps": steps, "error": f"Quantization failed: {quant_result['error']}"}
        
        param_path = quant_result["param_path"]
        bin_path = quant_result["bin_path"]
        print(f"✓ NCNN 转换成功:")
        print(f"  Param: {param_path}")
        print(f"  Bin: {bin_path}\n")
        
        if test_image and os.path.exists(test_image):
            print("="*50)
            print("Step 3: 推理测试")
            print("="*50)
            
            inference_result = self.inference(
                param_path=param_path,
                bin_path=bin_path,
                image_path=test_image,
                **inference_kwargs
            )
            steps["inference"] = inference_result
            
            if inference_result["success"]:
                print(f"✓ 推理成功: {len(inference_result['detections'])} 个检测框")
                print(f"  耗时: {inference_result['inference_time']:.2f} ms")
                if inference_result.get("output_image"):
                    print(f"  输出图像: {inference_result['output_image']}")
            else:
                print(f"✗ 推理失败: {inference_result['error']}")
        
        return {"success": True, "steps": steps, "error": None}
    
    # ==================== 工具函数 ====================
    
    def get_status(self) -> Dict:
        """获取当前环境状态"""
        env_check = self.check_environment()
        
        return {
            "base_dir": env_check["base_dir"],
            "python_dir": str(self.python_dir),
            "python_scripts": list(env_check["ready"].keys()),
            "cpp_built": "cpp_build" in env_check["ready"],
            "issues": env_check["issues"],
            "tools_dir": str(self.tools_dir) if self.tools_dir.exists() else None
        }


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Deploy - OpenCLAW Skill")
    parser.add_argument("command", choices=["export", "quantize", "inference", "full", "status"])
    parser.add_argument("--weights", help="Model weights path")
    parser.add_argument("--onnx", help="ONNX model path")
    parser.add_argument("--param", help="NCNN param path")
    parser.add_argument("--bin", help="NCNN bin path")
    parser.add_argument("--image", help="Test image path")
    parser.add_argument("--output", default="result.jpg", help="Output image path")
    parser.add_argument("--fp16", action="store_true", help="FP16 export")
    parser.add_argument("--quant", action="store_true", help="INT8 quantization")
    parser.add_argument("--calibration", "--calibration-images", dest="calibration", 
                        help="Calibration images directory")
    parser.add_argument("--ncnn-tools", help="NCNN tools path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (0-1)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold (0-1)")
    parser.add_argument("--threads", type=int, default=4, help="Thread count (1-32)")
    
    args = parser.parse_args()
    
    # 参数校验
    if args.conf and (args.conf <= 0 or args.conf > 1):
        print("Error: --conf must be in (0, 1]")
        return
    if args.iou and (args.iou <= 0 or args.iou > 1):
        print("Error: --iou must be in (0, 1]")
        return
    if args.threads and (args.threads < 1 or args.threads > 32):
        print("Error: --threads must be in [1, 32]")
        return
    
    deployer = YOLODeploy()
    
    if args.command == "status":
        status = deployer.get_status()
        print(json.dumps(status, indent=2))
        return
    
    if args.command == "export":
        if not args.weights:
            print("--weights required")
            return
        result = deployer.export_onnx(args.weights, fp16=args.fp16)
        print(json.dumps(result, indent=2))
    
    elif args.command == "quantize":
        if not args.onnx:
            print("--onnx required")
            return
        result = deployer.quantize(
            args.onnx,
            to_ncnn=True,
            quant=args.quant,
            calibration_images=args.calibration,
            ncnn_tools_path=args.ncnn_tools
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == "inference":
        if not args.param or not args.bin or not args.image:
            print("--param, --bin, --image required")
            return
        result = deployer.inference(
            args.param, args.bin, args.image,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            threads=args.threads
        )
        print(json.dumps(result, indent=2))
    
    elif args.command == "full":
        if not args.weights:
            print("--weights required")
            return
        result = deployer.full_pipeline(
            args.weights,
            test_image=args.image,
            fp16=args.fp16,
            quant=args.quant,
            calibration_images=args.calibration,
            ncnn_tools_path=args.ncnn_tools,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            threads=args.threads
        )
        
        if result["success"]:
            print("\n" + "="*50)
            print("✅ 完整流程执行成功!")
            print("="*50)
            print(f"  ONNX: {result['steps']['export'].get('onnx_path')}")
            print(f"  Param: {result['steps']['quantize'].get('param_path')}")
            print(f"  Bin: {result['steps']['quantize'].get('bin_path')}")
        else:
            print(f"\n❌ 流程失败: {result['error']}")


if __name__ == "__main__":
    main()