#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace glooms {
namespace vision {

// Forward declarations
class Processor;
class Detector;
class Analyzer;

// Common types and enums
enum class ProcessingMode {
    REALTIME,
    QUALITY,
    CUSTOM
};

enum class DetectionType {
    OBJECT,
    FACE,
    POSE,
    CUSTOM
};

// Configuration structures
struct VisionConfig {
    // General settings
    bool use_gpu = true;
    int gpu_id = 0;
    ProcessingMode mode = ProcessingMode::REALTIME;

    // Frame settings
    int frame_width = 1920;
    int frame_height = 1080;
    int target_fps = 30;

    // Processing settings
    bool enable_preprocessing = true;
    bool enable_detection = true;
    bool enable_analysis = true;
    
    // Model paths
    std::string detector_model;
    std::string analyzer_model;
    
    // Advanced settings
    bool enable_threading = true;
    int thread_count = 4;
    bool enable_logging = true;
};

// Result structures
struct ProcessedFrame {
    cv::Mat frame;
    uint64_t frame_number;
    double processing_time;
    std::vector<cv::Rect> regions;
    std::vector<std::string> labels;
    std::vector<float> confidences;
};

struct VisionResult {
    bool success;
    std::string message;
    ProcessedFrame frame;
    std::vector<cv::Mat> debug_frames;
};

// Callback types
using FrameCallback = std::function<void(const ProcessedFrame&)>;
using ErrorCallback = std::function<void(const std::string&)>;

// Main Vision class
class Vision {
public:
    // Constructor & Destructor
    explicit Vision(const VisionConfig& config);
    ~Vision();

    // Delete copy constructor and assignment operator
    Vision(const Vision&) = delete;
    Vision& operator=(const Vision&) = delete;

    // Core methods
    bool initialize();
    void cleanup();
    VisionResult processFrame(const cv::Mat& frame);
    VisionResult processVideo(const std::string& video_path);
    VisionResult processCamera(int camera_id = 0);

    // Configuration methods
    void setConfig(const VisionConfig& config);
    const VisionConfig& getConfig() const { return config_; }

    // Callback registration
    void setFrameCallback(FrameCallback callback);
    void setErrorCallback(ErrorCallback callback);

    // Pipeline control
    void start();
    void stop();
    void pause();
    void resume();
    bool isRunning() const { return is_running_; }

    // Utility methods
    static bool isGPUAvailable();
    static std::vector<cv::Size> getSupportedResolutions();
    static std::string getVersionInfo();

protected:
    // Pipeline methods
    bool initializePipeline();
    void processPipeline(const cv::Mat& frame);
    void cleanupPipeline();

    // Helper methods
    bool validateFrame(const cv::Mat& frame) const;
    void updateMetrics(const VisionResult& result);

private:
    // Configuration
    VisionConfig config_;
    bool is_initialized_;
    bool is_running_;
    bool gpu_enabled_;

    // Pipeline components
    std::unique_ptr<Processor> processor_;
    std::unique_ptr<Detector> detector_;
    std::unique_ptr<Analyzer> analyzer_;

    // Callbacks
    FrameCallback frame_callback_;
    ErrorCallback error_callback_;

    // Processing state
    uint64_t frame_count_;
    double total_processing_time_;
    std::vector<double> processing_times_;

    // Thread management
    std::vector<std::thread> worker_threads_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool should_stop_;

    // Internal helper methods
    void initializeThreads();
    void workerThread();
    void cleanupThreads();
};

// Factory functions
std::unique_ptr<Vision> createVision(const VisionConfig& config);
std::unique_ptr<Vision> createVisionFromFile(const std::string& config_path);

// Utility functions
namespace utils {
    cv::Mat resizeFrame(const cv::Mat& frame, int width, int height);
    cv::Mat preprocessFrame(const cv::Mat& frame, const VisionConfig& config);
    void drawResults(cv::Mat& frame, const ProcessedFrame& results);
    void saveResults(const std::string& path, const VisionResult& results);
    VisionResult loadResults(const std::string& path);
}

} // namespace vision
} // namespace glooms

// Convenience typedefs
using Vision = glooms::vision::Vision;
using VisionConfig = glooms::vision::VisionConfig;
using VisionResult = glooms::vision::VisionResult;