#pragma once

#include <opencv2/core.hpp>
#include <opencv2/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <memory>
#include <vector>

namespace glooms {
namespace vision {

// Forward declarations
class Logger;

// Enums
enum class ProcessingMode {
    REALTIME,
    QUALITY,
    CUSTOM
};

// Configuration struct
struct ProcessorConfig {
    // Frame settings
    int frame_width = 1920;
    int frame_height = 1080;
    bool use_gpu = true;

    // Processing settings
    ProcessingMode processing_mode = ProcessingMode::REALTIME;
    bool enable_edge_detection = false;
    bool enable_contour_detection = false;
    bool enable_color_segmentation = false;
    bool enable_motion_detection = false;

    // Neural network settings
    std::string model_path;
    int model_input_width = 416;
    int model_input_height = 416;
    float confidence_threshold = 0.5f;

    // Color segmentation bounds
    cv::Scalar color_lower_bound;
    cv::Scalar color_upper_bound;

    // Advanced settings
    int buffer_size = 30;
    bool enable_threading = true;
    int thread_count = 4;
};

// Processing result struct
struct ProcessingResult {
    bool success;
    std::string message;
    cv::Mat processed_frame;
    std::vector<cv::Mat> detections;
    uint64_t frame_number;
};

// Metrics struct
struct ProcessorMetrics {
    uint64_t frame_count;
    bool gpu_enabled;
    int frame_width;
    int frame_height;
    int processing_mode;
};

class VisionProcessor {
public:
    // Constructor & Destructor
    explicit VisionProcessor(const ProcessorConfig& config);
    ~VisionProcessor();

    // Delete copy constructor and assignment operator
    VisionProcessor(const VisionProcessor&) = delete;
    VisionProcessor& operator=(const VisionProcessor&) = delete;

    // Core methods
    bool initialize();
    void cleanup();
    ProcessingResult processFrame(const cv::Mat& frame);

    // Configuration methods
    void setConfig(const ProcessorConfig& config);
    const ProcessorConfig& getConfig() const { return config_; }

    // Metrics and status
    ProcessorMetrics getMetrics() const;
    bool isInitialized() const { return is_initialized_; }
    bool isGPUEnabled() const { return gpu_enabled_; }

    // Utility methods
    static bool isGPUAvailable() {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    }

protected:
    // Processing pipeline methods
    void preprocessFrame(const cv::Mat& input, cv::Mat& output);
    void applyVisionPipeline(cv::Mat& frame);
    void runInference(const cv::Mat& frame, std::vector<cv::Mat>& detections);

    // Helper methods
    bool validateFrame(const cv::Mat& frame) const;
    void updateMetrics(const ProcessingResult& result);

private:
    // Configuration
    ProcessorConfig config_;
    bool is_initialized_;
    bool gpu_enabled_;

    // OpenCV objects
    cv::Mat frame_buffer_;
    cv::Mat prev_frame_;
    cv::dnn::Net net_;
    cv::cuda::Stream gpu_stream_;

    // Processing state
    uint64_t frame_count_;
    std::vector<cv::Mat> frame_history_;

    // Utilities
    Logger& logger_;

    // Internal helper methods
    void initializeGPU();
    void initializeNetwork();
    void cleanupResources();
};

// Factory function
std::unique_ptr<VisionProcessor> createVisionProcessor(const ProcessorConfig& config);

} // namespace vision
} // namespace glooms