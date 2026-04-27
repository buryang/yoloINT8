#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>

struct Object {
    float x, y, w, h;
    float prob;
    int label;
};

const std::vector<std::string> COCO_CLASSES = {
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
};

class YOLO {
public:
    ncnn::Net net;
    int img_size = 640;
    int num_classes = 80;
    float nms_threshold = 0.45f;
    float conf_threshold = 0.25f;

    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    std::vector<float> norm = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};

    std::vector<int> strides = {8, 16, 32};

    bool verbose = false;
    int num_threads = 4;

    std::vector<std::string> input_names_cache;
    std::vector<std::string> output_names_cache;
    bool names_cached = false;

    ncnn::Extractor* extractor = nullptr;

    cv::Mat letterbox_buffer;

    int init(const std::string& param_path, const std::string& bin_path, bool use_quant = false) {
        net.opt.use_vulkan_compute = false;
        net.opt.num_threads = num_threads;

        if (verbose) {
            net.opt.verbose = true;
        }

        if (use_quant) {
            net.opt.use_fp16_packed = true;
            net.opt.use_int8_storage = true;
            net.opt.use_tf8_storage = true;
            net.opt.use_int8_arithmetic = true;
        } else {
            net.opt.use_fp16_packed = false;
            net.opt.use_int8_storage = false;
            net.opt.use_tf8_storage = false;
        }

        int ret = net.load_param(param_path.c_str());
        if (ret != 0) {
            std::cerr << "[ERROR] Failed to load param: " << param_path << std::endl;
            return -1;
        }

        ret = net.load_model(bin_path.c_str());
        if (ret != 0) {
            std::cerr << "[ERROR] Failed to load model: " << bin_path << std::endl;
            return -1;
        }

        if (net.input_names().size() > 0) {
            input_names_cache = net.input_names();
        } else {
            input_names_cache = {"images"};
        }
        
        if (net.output_names().size() > 0) {
            output_names_cache = net.output_names();
        } else {
            output_names_cache = {"output"};
        }
        names_cached = true;

        std::cout << "[INFO] NCNN model loaded (input: " << input_names_cache[0] 
                  << ", output: " << output_names_cache[0] << ")" << std::endl;
        return 0;
    }

    cv::Mat letterbox_resize(const cv::Mat& img, float& scale, int& pad_left, int& pad_top) {
        int w = img.cols;
        int h = img.rows;

        float scale_w = (float)img_size / w;
        float scale_h = (float)img_size / h;
        scale = std::min(scale_w, scale_h);

        int new_w = (int)(w * scale);
        int new_h = (int)(h * scale);

        pad_left = (img_size - new_w) / 2;
        pad_top = (img_size - new_h) / 2;

        if (letterbox_buffer.empty() || letterbox_buffer.cols != img_size || letterbox_buffer.rows != img_size) {
            letterbox_buffer = cv::Mat(img_size, img_size, CV_8UC3, cv::Scalar(114, 114, 114));
        } else {
            letterbox_buffer.setTo(cv::Scalar(114, 114, 114));
        }

        cv::Mat roi = letterbox_buffer(cv::Rect(pad_left, pad_top, new_w, new_h));
        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(new_w, new_h));
        resized_img.copyTo(roi);

        return letterbox_buffer;
    }

    ncnn::Mat preprocess(const cv::Mat& img) {
        ncnn::Mat in = ncnn::Mat::from_pixels(
            img.data, ncnn::Mat::PIXEL_BGR2RGB,
            img.cols, img.rows
        );
        in.substract_mean_normalize(mean.data(), norm.data());
        return in;
    }

    std::vector<Object> detect(const cv::Mat& img) {
        float scale;
        int pad_left, pad_top;
        cv::Mat resized = letterbox_resize(img, scale, pad_left, pad_top);

        ncnn::Mat in = preprocess(resized);

        if (!extractor) {
            extractor = net.create_extractor();
        }

        extractor->input(input_names_cache[0].c_str(), in);

        ncnn::Mat out;
        int ret = extractor->extract(output_names_cache[0].c_str(), out);

        if (ret != 0) {
            std::cerr << "[ERROR] Extract failed with code: " << ret << std::endl;
            return {};
        }

        return decode_outputs(out, img.cols, img.rows, scale, pad_left, pad_top);
    }

    std::vector<Object> decode_outputs_dynamic(const ncnn::Mat& out, int img_w, int img_h, float scale, int pad_left, int pad_top) {
        std::vector<Object> proposals;

        float* data = (float*)out.data;
        int c = out.c;
        int h = out.h;
        int w = out.w;
        int total_elements = c * h * w;

        if (total_elements < 85) {
            std::cerr << "[ERROR] Output too small: " << total_elements << " elements" << std::endl;
            return proposals;
        }

        int num_classes_detected = c - 4;
        if (num_classes_detected <= 0) {
            std::cerr << "[ERROR] Invalid channel count: " << c << std::endl;
            return proposals;
        }

        if (num_classes_detected > num_classes) {
            num_classes_detected = num_classes;
        }

        if (verbose) {
            std::cout << "[DEBUG] Output shape: c=" << c << ", h=" << h << ", w=" << w 
                      << ", detect_classes=" << num_classes_detected << std::endl;
        }

        int num_anchors = h * w;
        
        for (int i = 0; i < num_anchors; i++) {
            int base_idx = i * c;

            if (base_idx + c > total_elements) break;

            float conf = data[base_idx + 4];
            if (conf < conf_threshold) continue;

            float box_x = data[base_idx + 0];
            float box_y = data[base_idx + 1];
            float box_w = data[base_idx + 2];
            float box_h = data[base_idx + 3];

            if (box_w <= 0 || box_h <= 0) continue;

            if (box_x < 0 || box_y < 0 || box_x > img_size || box_y > img_size) continue;

            float best_prob = 0.0f;
            int best_class = -1;

            for (int cls = 0; cls < num_classes_detected; cls++) {
                float cls_prob = data[base_idx + 5 + cls] * conf;
                if (cls_prob > best_prob) {
                    best_prob = cls_prob;
                    best_class = cls;
                }
            }

            if (best_prob < conf_threshold || best_class < 0) continue;

            Object obj;
            float x = (box_x - pad_left) / scale;
            float y = (box_y - pad_top) / scale;
            float bw = box_w / scale;
            float bh = box_h / scale;
            
            obj.x = std::max(0.0f, x - bw / 2.0f);
            obj.y = std::max(0.0f, y - bh / 2.0f);
            obj.w = std::min((float)img_w - obj.x, bw);
            obj.h = std::min((float)img_h - obj.y, bh);

            if (obj.w <= 1 || obj.h <= 1) continue;

            obj.prob = best_prob;
            obj.label = best_class;
            proposals.push_back(obj);
        }

        return nms(proposals);
    }

    std::vector<Object> decode_outputs(const ncnn::Mat& out, int img_w, int img_h, float scale, int pad_left, int pad_top) {
        return decode_outputs_dynamic(out, img_w, img_h, scale, pad_left, pad_top);
    }

    std::vector<Object> nms(std::vector<Object>& objects) {
        if (objects.empty()) return objects;

        std::sort(objects.begin(), objects.end(),
            [](const Object& a, const Object& b) { return a.prob > b.prob; });

        std::unordered_map<int, std::vector<Object>> class_groups;
        for (const auto& obj : objects) {
            if (obj.prob >= conf_threshold) {
                class_groups[obj.label].push_back(obj);
            }
        }

        std::vector<Object> result;
        
        for (auto& pair : class_groups) {
            auto& class_objects = pair.second;
            
            std::sort(class_objects.begin(), class_objects.end(),
                [](const Object& a, const Object& b) { return a.prob > b.prob; });

            std::vector<bool> suppressed(class_objects.size(), false);
            
            for (size_t i = 0; i < class_objects.size(); i++) {
                if (suppressed[i]) continue;
                
                for (size_t j = i + 1; j < class_objects.size(); j++) {
                    if (suppressed[j]) continue;
                    
                    float iou_val = compute_iou(class_objects[i], class_objects[j]);
                    if (iou_val > nms_threshold) {
                        suppressed[j] = true;
                    }
                }
            }

            for (size_t i = 0; i < class_objects.size(); i++) {
                if (!suppressed[i]) result.push_back(class_objects[i]);
            }
        }

        std::sort(result.begin(), result.end(),
            [](const Object& a, const Object& b) { return a.prob > b.prob; });
        
        return result;
    }

    float compute_iou(const Object& a, const Object& b) {
        float x1 = std::max(a.x, b.x);
        float y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x + a.w, b.x + b.w);
        float y2 = std::min(a.y + a.h, b.y + b.h);

        float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
        float area_a = a.w * a.h;
        float area_b = b.w * b.h;
        float union_area = area_a + area_b - inter;

        return inter / (union_area + 1e-6f);
    }

    void draw_results(cv::Mat& img, const std::vector<Object>& objects, const std::string& output_path) {
        cv::Mat result = img.clone();
        
        for (const auto& obj : objects) {
            int x1 = (int)obj.x;
            int y1 = (int)obj.y;
            int x2 = (int)(obj.x + obj.w);
            int y2 = (int)(obj.y + obj.h);
            
            x1 = std::max(0, std::min(x1, result.cols - 1));
            y1 = std::max(0, std::min(y1, result.rows - 1));
            x2 = std::max(0, std::min(x2, result.cols - 1));
            y2 = std::max(0, std::min(y2, result.rows - 1));

            cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

            std::string label;
            if (obj.label >= 0 && obj.label < (int)COCO_CLASSES.size()) {
                label = COCO_CLASSES[obj.label];
            } else {
                label = "class_" + std::to_string(obj.label);
            }
            label += " " + std::to_string((int)(obj.prob * 100)) + "%";

            cv::putText(result, label, cv::Point(x1, y1 - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        cv::imwrite(output_path, result);
        std::cout << "[INFO] Result saved to: " << output_path << std::endl;
    }

    void destroy() {
        if (extractor) {
            delete extractor;
            extractor = nullptr;
        }
        net.clear();
    }
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " <param.bin> <model.bin> <image.jpg> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --quant           Use quantized model (int8)" << std::endl;
    std::cout << "  -o <path>         Output image path (default: result.jpg)" << std::endl;
    std::cout << "  -conf <float>     Confidence threshold (default: 0.25)" << std::endl;
    std::cout << "  -iou <float>      NMS IoU threshold (default: 0.45)" << std::endl;
    std::cout << "  -v, --verbose     Enable verbose logging" << std::endl;
    std::cout << "  -t, --threads N   Number of threads (default: 4)" << std::endl;
    std::cout << "Example: " << prog << " yolo.param yolo.bin test.jpg --quant -o result.jpg -conf 0.3 -iou 0.5" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return -1;
    }

    std::string param_path, bin_path, image_path, output_path = "result.jpg";
    bool use_quant = false;

    param_path = argv[1];
    bin_path = argv[2];
    image_path = argv[3];

    if (!std::ifstream(param_path).good()) {
        std::cerr << "[ERROR] Param file not found: " << param_path << std::endl;
        return -1;
    }
    if (!std::ifstream(bin_path).good()) {
        std::cerr << "[ERROR] Model file not found: " << bin_path << std::endl;
        return -1;
    }

    YOLO yolo;
    float default_conf = 0.25f;
    float default_iou = 0.45f;

    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--quant") {
            use_quant = true;
        } else if (arg == "-o" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "-conf" && i + 1 < argc) {
            default_conf = std::stof(argv[++i]);
        } else if (arg == "-iou" && i + 1 < argc) {
            default_iou = std::stof(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            yolo.verbose = true;
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            yolo.num_threads = std::stoi(argv[++i]);
        }
    }

    if (default_conf <= 0.0f || default_conf > 1.0f) {
        std::cerr << "[ERROR] -conf must be in (0, 1], got " << default_conf << std::endl;
        return -1;
    }
    if (default_iou <= 0.0f || default_iou > 1.0f) {
        std::cerr << "[ERROR] -iou must be in (0, 1], got " << default_iou << std::endl;
        return -1;
    }
    if (yolo.num_threads < 1 || yolo.num_threads > 32) {
        std::cerr << "[ERROR] -t/--threads must be in [1, 32], got " << yolo.num_threads << std::endl;
        return -1;
    }

    yolo.conf_threshold = default_conf;
    yolo.nms_threshold = default_iou;

    std::cout << "[INFO] Config: conf=" << default_conf << ", iou=" << default_iou 
              << ", quant=" << (use_quant ? "yes" : "no") << ", threads=" << yolo.num_threads << std::endl;

    if (yolo.init(param_path, bin_path, use_quant) != 0) {
        std::cerr << "[ERROR] Failed to initialize YOLO" << std::endl;
        return -1;
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << image_path << std::endl;
        return -1;
    }

    std::cout << "[INFO] Running inference..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Object> results = yolo.detect(img);
    auto end = std::chrono::high_resolution_clock::now();

    float infer_time = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << "[INFO] Inference time: " << infer_time << " ms" << std::endl;

    std::cout << "[INFO] Detected " << results.size() << " objects:" << std::endl;
    for (const auto& obj : results) {
        std::string class_name = (obj.label >= 0 && obj.label < (int)COCO_CLASSES.size()) 
            ? COCO_CLASSES[obj.label] : "class_" + std::to_string(obj.label);
        std::cout << "  [" << class_name << "] " << obj.prob * 100 << "% "
                  << "bbox=[" << obj.x << ", " << obj.y << ", " << obj.w << ", " << obj.h << "]" << std::endl;
    }

    yolo.draw_results(img, results, output_path);

    yolo.destroy();

    std::cout << "[INFO] Done!" << std::endl;
    return 0;
}