#include "utils/logger.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <mutex>

namespace glooms {
namespace utils {

// ANSI color codes
namespace Color {
    const char* RESET   = "\033[0m";
    const char* RED     = "\033[31m";
    const char* GREEN   = "\033[32m";
    const char* YELLOW  = "\033[33m";
    const char* BLUE    = "\033[34m";
    const char* MAGENTA = "\033[35m";
    const char* CYAN    = "\033[36m";
    const char* WHITE   = "\033[37m";
}

// Static member initialization
std::mutex Logger::mutex_;
LogLevel Logger::global_level_ = LogLevel::INFO;
std::shared_ptr<std::ostream> Logger::output_stream_ = std::make_shared<std::ostream>(std::cout.rdbuf());
bool Logger::use_colors_ = true;
std::string Logger::time_format_ = "%Y-%m-%d %H:%M:%S";

Logger::Logger(const std::string& prefix)
    : prefix_(prefix)
    , level_(global_level_)
    , enabled_(true) {}

void Logger::setLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

void Logger::setGlobalLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    global_level_ = level;
}

void Logger::setOutputStream(std::shared_ptr<std::ostream> stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    output_stream_ = stream;
}

void Logger::setUseColors(bool use_colors) {
    std::lock_guard<std::mutex> lock(mutex_);
    use_colors_ = use_colors;
}

void Logger::setTimeFormat(const std::string& format) {
    std::lock_guard<std::mutex> lock(mutex_);
    time_format_ = format;
}

void Logger::log(LogLevel level, const std::string& message, const LogContext& context) {
    if (!enabled_ || level < level_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    
    std::stringstream ss;
    
    // Add timestamp
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()
    ).count() % 1000;
    
    ss << std::put_time(std::localtime(&time), time_format_.c_str())
       << "." << std::setfill('0') << std::setw(3) << ms << " ";

    // Add level with color
    if (use_colors_) {
        switch (level) {
            case LogLevel::TRACE:   ss << Color::WHITE   << "TRACE"; break;
            case LogLevel::DEBUG:   ss << Color::BLUE    << "DEBUG"; break;
            case LogLevel::INFO:    ss << Color::GREEN   << "INFO "; break;
            case LogLevel::WARN:    ss << Color::YELLOW  << "WARN "; break;
            case LogLevel::ERROR:   ss << Color::RED     << "ERROR"; break;
            case LogLevel::FATAL:   ss << Color::MAGENTA << "FATAL"; break;
        }
        ss << Color::RESET;
    } else {
        switch (level) {
            case LogLevel::TRACE:   ss << "TRACE"; break;
            case LogLevel::DEBUG:   ss << "DEBUG"; break;
            case LogLevel::INFO:    ss << "INFO "; break;
            case LogLevel::WARN:    ss << "WARN "; break;
            case LogLevel::ERROR:   ss << "ERROR"; break;
            case LogLevel::FATAL:   ss << "FATAL"; break;
        }
    }

    // Add prefix and message
    ss << " [" << prefix_ << "] " << message;

    // Add context if available
    if (!context.empty()) {
        ss << " {";
        bool first = true;
        for (const auto& [key, value] : context) {
            if (!first) ss << ", ";
            ss << key << ": " << value;
            first = false;
        }
        ss << "}";
    }

    ss << std::endl;

    // Write to output stream
    (*output_stream_) << ss.str();
    output_stream_->flush();
}

void Logger::trace(const std::string& message, const LogContext& context) {
    log(LogLevel::TRACE, message, context);
}

void Logger::debug(const std::string& message, const LogContext& context) {
    log(LogLevel::DEBUG, message, context);
}

void Logger::info(const std::string& message, const LogContext& context) {
    log(LogLevel::INFO, message, context);
}

void Logger::warn(const std::string& message, const LogContext& context) {
    log(LogLevel::WARN, message, context);
}

void Logger::error(const std::string& message, const LogContext& context) {
    log(LogLevel::ERROR, message, context);
}

void Logger::fatal(const std::string& message, const LogContext& context) {
    log(LogLevel::FATAL, message, context);
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default:              return "UNKNOWN";
    }
}

LogLevel Logger::stringToLevel(const std::string& level) {
    if (level == "TRACE") return LogLevel::TRACE;
    if (level == "DEBUG") return LogLevel::DEBUG;
    if (level == "INFO")  return LogLevel::INFO;
    if (level == "WARN")  return LogLevel::WARN;
    if (level == "ERROR") return LogLevel::ERROR;
    if (level == "FATAL") return LogLevel::FATAL;
    return LogLevel::INFO;
}

} // namespace utils
} // namespace glooms