#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

template<typename T>
T my_min(T a, T b) { return a < b ? a : b; }

template<typename T>
T my_max(T a, T b) { return a > b ? a : b; }

#include <opencv2/opencv.hpp>
#include <ncnn/net.h>

struct Detection {
    cv::Rect2d bbox;
    float confidence;
    int class_id;
};

const std::vector<std::string> CLASS_NAMES = {
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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <param> <bin> <image> [conf] [iou]" << std::endl;
        return -1;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    const char* image_path = argv[3];
    float conf_threshold = (argc > 4) ? atof(argv[4]) : 0.25f;
    float iou_threshold = (argc > 5) ? atof(argv[5]) : 0.45f;

    // Load model
    ncnn::Net net;
    if (net.load_param(param_path) != 0) {
        std::cerr << "Failed to load param: " << param_path << std::endl;
        return -1;
    }
    if (net.load_model(bin_path) != 0) {
        std::cerr << "Failed to load model: " << bin_path << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // Load image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "Image: " << img.cols << "x" << img.rows << std::endl;

    // Preprocess: letterbox resize to 640x640 with pad value 114
    int img_size = 640;
    int h = img.rows, w = img.cols;
    float scale = (float)img_size / my_max(h, w);
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    cv::Mat padded = cv::Mat::zeros(img_size, img_size, CV_8UC3);
    padded.setTo(cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    // BGR to RGB
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);

    // Convert to ncnn::Mat and normalize by /255
    ncnn::Mat in = ncnn::Mat::from_pixels(
        padded.data, ncnn::Mat::PIXEL_RGB, img_size, img_size
    );

    static const float norm[3] = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
    in.substract_mean_normalize(0, norm);

    // Inference
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat out;
    ex.extract("out0", out);

    const float* ptr = (const float*)out.data;
    int W = out.w;
    int H = out.h;
    int C = out.c;
    
std::vector<Detection> detections;
    int num_anchors = W;
    int num_channels = 84;
    int debug_count = 0;

    // Use layout: ptr[channel * W + anchor]
    for (int i = 0; i < num_anchors; i++) {
        // Model outputs (cx, cy, w, h) in 640-space
        float cx = ptr[0 * W + i];
        float cy = ptr[1 * W + i];
        float bw = ptr[2 * W + i];
        float bh = ptr[3 * W + i];

        // Convert to xyxy format
        float x1 = cx - bw * 0.5f;
        float y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f;
        float y2 = cy + bh * 0.5f;

// Find max class - handle both raw logits and sigmoid'd values
        int max_class = 0;
        float max_conf = 0;
        for (int c = 0; c < 80; c++) {
            float conf = ptr[(4 + c) * W + i];
            // If value > 1, it's raw logit - apply sigmoid
            // If value <= 1, it's already sigmoid'd probability
            if (conf > 1.0f) {
                conf = 1.0f / (1.0f + exp(-conf));
            }
            if (conf > max_conf) {
                max_conf = conf;
                max_class = c;
            }
        }

        if (max_conf > conf_threshold) {
            // Debug
            if (debug_count < 3) {
                std::cout << "Anchor " << i << ": bbox=(" << x1 << "," << y1 << "," << x2 << "," << y2 
                          << ") orig=(" << (x1/scale) << "," << (y1/scale) << "," << (x2/scale) << "," << (y2/scale) << ")"
                          << " conf=" << max_conf << " class=" << max_class << std::endl;
                debug_count++;
            }
        
            // Convert to original image coords
            float ox1 = x1 / scale;
            float oy1 = y1 / scale;
            float ox2 = x2 / scale;
            float oy2 = y2 / scale;

            // Validate bbox - check reasonable size and position
            if (ox2 > ox1 && oy2 > oy1 && ox2 - ox1 > 5 && oy2 - oy1 > 5 && 
                ox1 >= 0 && oy1 >= 0 && ox2 <= w && oy2 <= h) {
                // Clip
                ox1 = my_max(0.0f, my_min((float)w, ox1));
                oy1 = my_max(0.0f, my_min((float)h, oy1));
                ox2 = my_max(0.0f, my_min((float)w, ox2));
                oy2 = my_max(0.0f, my_min((float)h, oy2));

                Detection det;
                det.class_id = max_class;
                det.confidence = max_conf;
                det.bbox = cv::Rect2d(ox1, oy1, ox2 - ox1, oy2 - oy1);
                detections.push_back(det);
            }
        }
    }

    std::cout << "Detections before NMS: " << detections.size() << std::endl;

    // NMS
    std::sort(detections.begin(), detections.end(), 
        [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(detections.size(), false);
    std::vector<Detection> result;

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (detections[i].class_id != detections[j].class_id) continue;

            cv::Rect2d a = detections[i].bbox;
            cv::Rect2d b = detections[j].bbox;
            float x1 = my_max(a.x, b.x);
            float y1 = my_max(a.y, b.y);
            float x2 = my_min(a.x + a.width, b.x + b.width);
            float y2 = my_min(a.y + a.height, b.y + b.height);
            float inter = my_max(0.0f, x2 - x1) * my_max(0.0f, y2 - y1);
            float area1 = a.width * a.height;
            float area2 = b.width * b.height;
            float iou = inter / (area1 + area2 - inter + 1e-6);

            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    std::cout << "\n=== Detection Results ===" << std::endl;
    std::cout << "Detections: " << result.size() << std::endl;

    for (const auto& det : result) {
        std::cout << CLASS_NAMES[det.class_id] << ": " 
                  << int(det.confidence * 100) << "% "
                  << "bbox=[" << int(det.bbox.x) << "," << int(det.bbox.y) 
                  << "," << int(det.bbox.width) << "," << int(det.bbox.height) << "]" 
                  << std::endl;
        
        cv::rectangle(img, det.bbox, cv::Scalar(0, 255, 0), 2);
        std::string label = CLASS_NAMES[det.class_id] + " " + std::to_string(int(det.confidence * 100)) + "%";
        cv::putText(img, label, cv::Point(det.bbox.x, det.bbox.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::imwrite("result.jpg", img);
    std::cout << "\nResult saved to result.jpg" << std::endl;

    return 0;
}