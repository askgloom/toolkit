#pragma once

#include <string>
#include <memory>
#include <mutex>
#include <map>
#include <ostream>

namespace glooms {
namespace utils {

// Enums
enum class LogLevel {
    TRACE,
    DEBUG,
    INFO,
    WARN,
    ERROR,
    FATAL
};

// Type definitions
using LogContext = std::map<std::string, std::string>;

class Logger {
public:
    // Constructor
    explicit Logger(const std::string& prefix);

    // Destructor
    ~Logger() = default;

    // Delete copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    // Core logging methods
    void trace(const std::string& message, const LogContext& context = {});
    void debug(const std::string& message, const LogContext& context = {});
    void info(const std::string& message, const LogContext& context = {});
    void warn(const std::string& message, const LogContext& context = {});
    void error(const std::string& message, const LogContext& context = {});
    void fatal(const std::string& message, const LogContext& context = {});

    // Configuration methods
    void setLevel(LogLevel level);
    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }
    bool isEnabled() const { return enabled_; }
    LogLevel getLevel() const { return level_; }

    // Static configuration methods
    static void setGlobalLevel(LogLevel level);
    static void setOutputStream(std::shared_ptr<std::ostream> stream);
    static void setUseColors(bool use_colors);
    static void setTimeFormat(const std::string& format);

    // Utility methods
    static std::string levelToString(LogLevel level);
    static LogLevel stringToLevel(const std::string& level);

    // Macro helpers
    static constexpr const char* getSourceLocation(const char* file) {
        const char* last = file;
        for (const char* ptr = file; *ptr; ++ptr) {
            if (*ptr == '/' || *ptr == '\\') {
                last = ptr + 1;
            }
        }
        return last;
    }

protected:
    // Core logging implementation
    void log(LogLevel level, const std::string& message, const LogContext& context = {});

private:
    // Instance members
    std::string prefix_;
    LogLevel level_;
    bool enabled_;

    // Static members
    static std::mutex mutex_;
    static LogLevel global_level_;
    static std::shared_ptr<std::ostream> output_stream_;
    static bool use_colors_;
    static std::string time_format_;
};

// Convenience macros
#define LOG_TRACE(logger, message, ...) \
    logger.trace(message, ##__VA_ARGS__)

#define LOG_DEBUG(logger, message, ...) \
    logger.debug(message, ##__VA_ARGS__)

#define LOG_INFO(logger, message, ...) \
    logger.info(message, ##__VA_ARGS__)

#define LOG_WARN(logger, message, ...) \
    logger.warn(message, ##__VA_ARGS__)

#define LOG_ERROR(logger, message, ...) \
    logger.error(message, ##__VA_ARGS__)

#define LOG_FATAL(logger, message, ...) \
    logger.fatal(message, ##__VA_ARGS__)

// Source location macros
#define LOG_LOCATION \
    std::string(__FILE__) + ":" + std::to_string(__LINE__)

#define LOG_CONTEXT(logger, message, context) \
    { \
        auto ctx = context; \
        ctx["location"] = LOG_LOCATION; \
        logger.log(message, ctx); \
    }

// Scoped logging
class ScopedLogger {
public:
    ScopedLogger(Logger& logger, const std::string& scope)
        : logger_(logger)
        , scope_(scope) {
        logger_.trace("Entering " + scope_);
    }

    ~ScopedLogger() {
        logger_.trace("Exiting " + scope_);
    }

private:
    Logger& logger_;
    std::string scope_;
};

#define SCOPED_LOG(logger, scope) \
    ScopedLogger scoped_logger##__LINE__(logger, scope)

// Builder pattern for context
class LogContextBuilder {
public:
    LogContextBuilder& add(const std::string& key, const std::string& value) {
        context_[key] = value;
        return *this;
    }

    LogContext build() const {
        return context_;
    }

private:
    LogContext context_;
};

} // namespace utils
} // namespace glooms

// Convenience using directive
using Logger = glooms::utils::Logger;
using LogContextBuilder = glooms::utils::LogContextBuilder;