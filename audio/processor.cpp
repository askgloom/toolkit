#include "vision/processor.hpp"
#include "utils/logger.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <tensorflow/core/framework/tensor.h>

namespace glooms {
namespace vision {

Processor::Processor(const ProcessorConfig& config)
    : config_(config)
    , logger_("VisionProcessor")
    , frame_count_(0)
    , is_initialized_(false) {
    initialize();
}

Processor::~Processor() {
    cleanup();
}

bool Processor::initialize() {
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

        // Initialize preprocessing parameters
        resize_dims_ = cv::Size(config_.target_width, config_.target_height);
        mean_values_ = cv::Scalar(config_.mean[0], config_.mean[1], config_.mean[2]);
        std_values_ = cv::Scalar(config_.std[0], config_.std[1], config_.std[2]);

        is_initialized_ = true;
        logger_.info("Processor initialized successfully");
        return true;

    } catch (const std::exception& e) {
        logger_.error("Failed to initialize processor: " + std::string(e.what()));
        return false;
    }
}

void Processor::cleanup() {
    if (gpu_enabled_) {
        gpu_stream_.waitForCompletion();
    }
    frame_buffer_.release();
    is_initialized_ = false;
    logger_.info("Processor cleanup completed");
}

ProcessingResult Processor::processFrame(const cv::Mat& frame) {
    if (!is_initialized_) {
        return ProcessingResult{false, "Processor not initialized"};
    }

    try {
        cv::Mat processed;
        frame_count_++;

        // Basic preprocessing
        preprocessFrame(frame, processed);

        // Color space conversion if needed
        if (config_.color_space != ColorSpace::BGR) {
            convertColorSpace(processed);
        }

        // Apply additional processing based on configuration
        if (config_.enable_noise_reduction) {
            applyNoiseReduction(processed);
        }

        if (config_.enable_contrast_enhancement) {
            enhanceContrast(processed);
        }

        // Prepare tensor for ML model
        tensorflow::Tensor tensor;
        if (!prepareInputTensor(processed, tensor)) {
            return ProcessingResult{false, "Failed to prepare input tensor"};
        }

        return ProcessingResult{
            true,
            "Frame processed successfully",
            processed,
            tensor,
            frame_count_
        };

    } catch (const std::exception& e) {
        logger_.error("Frame processing failed: " + std::string(e.what()));
        return ProcessingResult{false, "Processing error: " + std::string(e.what())};
    }
}

void Processor::preprocessFrame(const cv::Mat& input, cv::Mat& output) {
    if (gpu_enabled_) {
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(input, gpu_stream_);

        // Resize
        cv::cuda::GpuMat gpu_resized;
        cv::cuda::resize(gpu_frame, gpu_resized, resize_dims_, 0, 0, cv::INTER_LINEAR, gpu_stream_);

        // Normalize
        gpu_resized.convertTo(gpu_resized, CV_32F, 1.0/255.0, gpu_stream_);
        
        // Download result
        gpu_resized.download(output, gpu_stream_);
        gpu_stream_.waitForCompletion();

    } else {
        // CPU processing path
        cv::resize(input, output, resize_dims_, 0, 0, cv::INTER_LINEAR);
        output.convertTo(output, CV_32F, 1.0/255.0);
    }

    // Normalize using mean and std
    cv::subtract(output, mean_values_, output);
    cv::divide(output, std_values_, output);
}

void Processor::convertColorSpace(cv::Mat& frame) {
    switch (config_.color_space) {
        case ColorSpace::RGB:
            cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
            break;
        case ColorSpace::HSV:
            cv::cvtColor(frame, frame, cv::COLOR_BGR2HSV);
            break;
        case ColorSpace::LAB:
            cv::cvtColor(frame, frame, cv::COLOR_BGR2Lab);
            break;
        default:
            break; // Keep BGR
    }
}

void Processor::applyNoiseReduction(cv::Mat& frame) {
    if (gpu_enabled_) {
        cv::cuda::GpuMat gpu_frame;
        gpu_frame.upload(frame, gpu_stream_);
        cv::cuda::fastNlMeansDenoisingColored(gpu_frame, gpu_frame, 
            config_.noise_h, config_.noise_template_size, 
            config_.noise_search_size, gpu_stream_);
        gpu_frame.download(frame, gpu_stream_);
        gpu_stream_.waitForCompletion();
    } else {
        cv::fastNlMeansDenoisingColored(frame, frame,
            config_.noise_h, config_.noise_template_size,
            config_.noise_search_size);
    }
}

void Processor::enhanceContrast(cv::Mat& frame) {
    cv::Mat ycrcb;
    cv::cvtColor(frame, ycrcb, cv::COLOR_BGR2YCrCb);
    
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    
    cv::equalizeHist(channels[0], channels[0]);
    
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, frame, cv::COLOR_YCrCb2BGR);
}

bool Processor::prepareInputTensor(const cv::Mat& frame, tensorflow::Tensor& tensor) {
    try {
        // Create tensor of shape {1, height, width, channels}
        tensorflow::TensorShape shape({1, 
            config_.target_height, 
            config_.target_width, 
            frame.channels()
        });
        
        tensor = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
        
        // Copy data from cv::Mat to tensor
        auto tensor_mapped = tensor.tensor<float, 4>();
        const float* frame_data = (float*)frame.data;
        
        for (int y = 0; y < frame.rows; ++y) {
            for (int x = 0; x < frame.cols; ++x) {
                for (int c = 0; c < frame.channels(); ++c) {
                    tensor_mapped(0, y, x, c) = frame_data[
                        y * frame.cols * frame.channels() + 
                        x * frame.channels() + c
                    ];
                }
            }
        }
        
        return true;

    } catch (const std::exception& e) {
        logger_.error("Tensor preparation failed: " + std::string(e.what()));
        return false;
    }
}

ProcessorMetrics Processor::getMetrics() const {
    return ProcessorMetrics{
        frame_count_,
        gpu_enabled_,
        config_.target_width,
        config_.target_height,
        static_cast<int>(config_.color_space)
    };
}

} // namespace vision
} // namespace glooms