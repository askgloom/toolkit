#include "vision/detector.hpp"
#include "utils/logger.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <fstream>
#include <algorithm>

namespace glooms {
namespace vision {

Detector::Detector(const DetectorConfig& config)
    : config_(config)
    , logger_("VisionDetector")
    , detection_count_(0)
    , is_initialized_(false) {
    initialize();
}

Detector::~Detector() {
    cleanup();
}

bool Detector::initialize() {
    try {
        // Load class names
        std::ifstream ifs(config_.classes_file);
        if (!ifs.is_open()) {
            logger_.error("Failed to load class names file: " + config_.classes_file);
            return false;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            class_names_.push_back(line);
        }

        // Load neural network
        net_ = cv::dnn::readNet(config_.model_weights, config_.model_config);
        
        // Configure backend
        if (config_.use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            gpu_enabled_ = true;
            logger_.info("GPU acceleration enabled for detection");
        } else {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            gpu_enabled_ = false;
            logger_.warn("Using CPU for detection");
        }

        // Initialize output layer names
        output_names_ = net_.getUnconnectedOutLayersNames();

        is_initialized_ = true;
        logger_.info("Detector initialized successfully");
        return true;

    } catch (const std::exception& e) {
        logger_.error("Failed to initialize detector: " + std::string(e.what()));
        return false;
    }
}

void Detector::cleanup() {
    net_.clear();
    class_names_.clear();
    is_initialized_ = false;
    logger_.info("Detector cleanup completed");
}

DetectionResult Detector::detect(const cv::Mat& frame) {
    if (!is_initialized_) {
        return DetectionResult{false, "Detector not initialized"};
    }

    try {
        detection_count_++;
        std::vector<Detection> detections;

        // Create blob from image
        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 
            1/255.0,
            cv::Size(config_.input_width, config_.input_height),
            cv::Scalar(0,0,0),
            true, 
            false
        );

        // Run inference
        net_.setInput(blob);
        std::vector<cv::Mat> outputs;
        net_.forward(outputs, output_names_);

        // Process detections
        processDetections(frame, outputs, detections);

        // Apply non-maximum suppression if enabled
        if (config_.enable_nms) {
            applyNMS(detections);
        }

        return DetectionResult{
            true,
            "Detection successful",
            detections,
            detection_count_
        };

    } catch (const std::exception& e) {
        logger_.error("Detection failed: " + std::string(e.what()));
        return DetectionResult{false, "Detection error: " + std::string(e.what())};
    }
}

void Detector::processDetections(
    const cv::Mat& frame,
    const std::vector<cv::Mat>& outputs,
    std::vector<Detection>& detections
) {
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // Get frame dimensions
    float width_ratio = float(frame.cols) / config_.input_width;
    float height_ratio = float(frame.rows) / config_.input_height;

    // Process network outputs
    for (const auto& output : outputs) {
        float* data = (float*)output.data;
        for (int i = 0; i < output.rows; ++i, data += output.cols) {
            cv::Mat scores = output.row(i).colRange(5, output.cols);
            cv::Point class_id_point;
            double confidence;
            
            // Get maximum score and its index
            cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id_point);
            
            if (confidence > config_.confidence_threshold) {
                // Get bounding box dimensions
                float center_x = data[0] * width_ratio;
                float center_y = data[1] * height_ratio;
                float width = data[2] * width_ratio;
                float height = data[3] * height_ratio;
                float x = center_x - width/2;
                float y = center_y - height/2;

                // Store detection
                Detection det;
                det.class_id = class_id_point.x;
                det.confidence = static_cast<float>(confidence);
                det.box = cv::Rect(x, y, width, height);
                det.class_name = class_names_[det.class_id];
                
                detections.push_back(det);
            }
        }
    }
}

void Detector::applyNMS(std::vector<Detection>& detections) {
    // Sort detections by confidence
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<Detection> selected;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;

        selected.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;

            float iou = calculateIoU(detections[i].box, detections[j].box);
            if (iou > config_.nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    detections = selected;
}

float Detector::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    float intersection_area = (x2 - x1) * (y2 - y1);
    float box1_area = box1.width * box1.height;
    float box2_area = box2.width * box2.height;
    float union_area = box1_area + box2_area - intersection_area;

    return intersection_area / union_area;
}

DetectorMetrics Detector::getMetrics() const {
    return DetectorMetrics{
        detection_count_,
        gpu_enabled_,
        static_cast<int>(class_names_.size()),
        config_.input_width,
        config_.input_height
    };
}

} // namespace vision
} // namespace glooms