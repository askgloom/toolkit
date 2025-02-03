#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <memory>

namespace glooms {
namespace vision {

// Forward declarations
class Logger;

// Structs and enums
struct Detection {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point> keypoints;  // Optional keypoints
    cv::Mat mask;                      // Optional segmentation mask
};

struct DetectorConfig {
    // Model settings
    std::string model_weights;
    std::string model_config;
    std::string classes_file;
    
    // Input settings
    int input_width = 416;
    int input_height = 416;
    bool maintain_aspect_ratio = true;

    // Detection settings
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    bool enable_nms = true;
    
    // Hardware settings
    bool use_gpu = true;
    int gpu_id = 0;

    // Advanced settings
    bool enable_batch_processing = false;
    int max_batch_size = 1;
    bool enable_keypoints = false;
    bool enable_segmentation = false;
};

struct DetectionResult {
    bool success;
    std::string message;
    std::vector<Detection> detections;
    uint64_t frame_number;
};

struct DetectorMetrics {
    uint64_t detection_count;
    bool gpu_enabled;
    int num_classes;
    int input_width;
    int input_height;
};

class Detector {
public:
    // Constructor & Destructor
    explicit Detector(const DetectorConfig& config);
    ~Detector();

    // Delete copy constructor and assignment operator
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // Core methods
    bool initialize();
    void cleanup();
    DetectionResult detect(const cv::Mat& frame);

    // Batch processing methods
    DetectionResult detectBatch(const std::vector<cv::Mat>& frames);
    
    // Configuration methods
    void setConfig(const DetectorConfig& config) { config_ = config; }
    const DetectorConfig& getConfig() const { return config_; }
    
    // Class management
    const std::vector<std::string>& getClassNames() const { return class_names_; }
    bool addClass(const std::string& class_name);
    bool removeClass(const std::string& class_name);

    // Metrics and status
    DetectorMetrics getMetrics() const;
    bool isInitialized() const { return is_initialized_; }
    bool isGPUEnabled() const { return gpu_enabled_; }

    // Utility methods
    static float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    static bool isGPUAvailable() { return cv::cuda::getCudaEnabledDeviceCount() > 0; }

protected:
    // Detection pipeline methods
    void processDetections(
        const cv::Mat& frame,
        const std::vector<cv::Mat>& outputs,
        std::vector<Detection>& detections
    );
    
    void applyNMS(std::vector<Detection>& detections);
    
    // Helper methods
    bool loadClasses();
    bool validateFrame(const cv::Mat& frame) const;
    void preprocessFrame(const cv::Mat& frame, cv::Mat& blob);
    void postprocessDetections(std::vector<Detection>& detections);

private:
    // Configuration
    DetectorConfig config_;
    bool is_initialized_;
    bool gpu_enabled_;

    // Neural network
    cv::dnn::Net net_;
    std::vector<std::string> output_names_;
    std::vector<std::string> class_names_;

    // Processing state
    uint64_t detection_count_;
    std::vector<cv::Mat> batch_buffer_;

    // Utilities
    Logger& logger_;

    // Internal helper methods
    void initializeGPU();
    void initializeNetwork();
    void cleanupResources();
};

// Factory function
std::unique_ptr<Detector> createDetector(const DetectorConfig& config);

// Utility functions
namespace utils {
    std::vector<cv::Scalar> generateColors(int num_classes);
    void drawDetections(cv::Mat& frame, const std::vector<Detection>& detections);
    void saveDetections(const std::string& filename, const std::vector<Detection>& detections);
    std::vector<Detection> loadDetections(const std::string& filename);
}

} // namespace vision
} // namespace glooms