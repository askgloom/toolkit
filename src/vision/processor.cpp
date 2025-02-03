#include "vision/processor.hpp"
#include "utils/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace glooms {
namespace vision {

VisionProcessor::VisionProcessor(const ProcessorConfig& config)
    : config_(config)
    , logger_("VisionProcessor")
    , frame_count_(0)
    , is_initialized_(false) {
    initialize();
}

VisionProcessor::~VisionProcessor() {
    cleanup();
}

bool VisionProcessor::initialize() {
    try {
        // Initialize frame buffer
        frame_buffer_ = cv::Mat(
            config_.frame_height,
            config_.frame_width,
            CV_8UC3
        );

        // Initialize GPU context if available
        if (config_.use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            gpu_stream_ = cv::cuda::Stream();
            gpu_enabled_ = true;
            logger_.info("GPU acceleration enabled");
        } else {
            gpu_enabled_ = false;
            logger_.warn("GPU acceleration not available");
        }

        // Load neural network model if specified
        if (!config_.model_path.empty()) {
            net_ = cv::dnn::readNet(config_.model_path);
            if (gpu_enabled_) {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
        }

        is_initialized_ = true;
        logger_.info("Vision processor initialized successfully");
        return true;

    } catch (const std::exception& e) {
        logger_.error("Failed to initialize vision processor: " + std::string(e.what()));
        return false;
    }
}

void VisionProcessor::cleanup() {
    if (gpu_enabled_) {
        gpu_stream_.waitForCompletion();
    }
    frame_buffer_.release();
    is_initialized_ = false;
    logger_.info("Vision processor cleanup completed");
}

ProcessingResult VisionProcessor::processFrame(const cv::Mat& frame) {
    if (!is_initialized_) {
        return ProcessingResult{false, "Processor not initialized"};
    }

    try {
        cv::Mat processed;
        frame_count_++;

        // Basic preprocessing
        preprocessFrame(frame, processed);

        // Apply vision processing pipeline
        applyVisionPipeline(processed);

        // Run neural network inference if model is loaded
        std::vector<cv::Mat> detections;
        if (!config_.model_path.empty()) {
            runInference(processed, detections);
        }

        return ProcessingResult{
            true,
            "Frame processed successfully",
            processed,
            detections,
            frame_count_
        };

    } catch (const std::exception& e) {
        logger_.error("Frame processing failed: " + std::string(e.what()));
        return ProcessingResult{false, "Processing error: " + std::string(e.what())};
    }
}

void VisionProcessor::preprocessFrame(const cv::Mat& input, cv::Mat& output) {
    if (gpu_enabled_) {
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(input, gpu_stream_);

        // Apply preprocessing on GPU
        cv::cuda::cvtColor(gpu_frame, gpu_frame, cv::COLOR_BGR2RGB, 0, gpu_stream_);
        cv::cuda::GaussianBlur(gpu_frame, gpu_frame, cv::Size(3, 3), 0, gpu_stream_);
        
        gpu_frame.download(output, gpu_stream_);
        gpu_stream_.waitForCompletion();
    } else {
        cv::cvtColor(input, output, cv::COLOR_BGR2RGB);
        cv::GaussianBlur(output, output, cv::Size(3, 3), 0);
    }
}

void VisionProcessor::applyVisionPipeline(cv::Mat& frame) {
    // Edge detection
    if (config_.enable_edge_detection) {
        cv::Mat edges;
        cv::Canny(frame, edges, 100, 200);
        cv::bitwise_and(frame, frame, frame, edges);
    }

    // Contour detection
    if (config_.enable_contour_detection) {
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0), 2);
    }

    // Color segmentation
    if (config_.enable_color_segmentation) {
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_RGB2HSV);
        cv::inRange(hsv, config_.color_lower_bound, config_.color_upper_bound, frame);
    }

    // Motion detection
    if (config_.enable_motion_detection && !prev_frame_.empty()) {
        cv::Mat diff;
        cv::absdiff(prev_frame_, frame, diff);
        cv::threshold(diff, diff, 25, 255, cv::THRESH_BINARY);
        frame = diff;
    }
    frame.copyTo(prev_frame_);
}

void VisionProcessor::runInference(const cv::Mat& frame, std::vector<cv::Mat>& detections) {
    // Prepare blob from image
    cv::Mat blob = cv::dnn::blobFromImage(
        frame,
        1.0,
        cv::Size(config_.model_input_width, config_.model_input_height),
        cv::Scalar(127.5, 127.5, 127.5),
        true,
        false
    );

    // Run forward pass
    net_.setInput(blob);
    detections = net_.forward();
}

ProcessorMetrics VisionProcessor::getMetrics() const {
    return ProcessorMetrics{
        frame_count_,
        gpu_enabled_,
        config_.frame_width,
        config_.frame_height,
        static_cast<int>(config_.processing_mode)
    };
}

void VisionProcessor::setConfig(const ProcessorConfig& config) {
    config_ = config;
    if (is_initialized_) {
        cleanup();
        initialize();
    }
}

} // namespace vision
} // namespace glooms