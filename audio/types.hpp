#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <optional>
#include <variant>
#include <unordered_map>
#include <memory>

namespace gloom {

// Forward declarations
class Agent;
class Memory;

// Basic types
using Vector = std::vector<float>;
using TimePoint = std::chrono::system_clock::time_point;
using AgentID = std::string;
using MemoryID = std::string;

// Configuration structures
struct MemoryConfig {
    size_t capacity = 1000;
    float decay_rate = 0.1f;
    float retrieval_threshold = 0.5f;
    size_t embed_dimension = 384;
    std::chrono::milliseconds consolidation_interval{3600000}; // 1 hour
    bool enable_persistence = false;
    std::string storage_path = "";
};

struct AgentConfig {
    std::string name;
    MemoryConfig memory;
    size_t max_tokens = 2048;
    float temperature = 0.7f;
    std::unordered_map<std::string, std::string> model_config;
};

// Memory structures
struct MemoryEntry {
    MemoryID id;
    std::string content;
    Vector embedding;
    TimePoint timestamp;
    float importance;
    std::string type;
    std::unordered_map<std::string, std::string> metadata;
    size_t access_count = 0;
    TimePoint last_accessed;

    // Serialization
    template<typename Archive>
    void serialize(Archive& ar) {
        ar(id, content, embedding, timestamp, importance,
           type, metadata, access_count, last_accessed);
    }
};

struct MemorySearchResult {
    MemoryID id;
    float similarity;
    MemoryEntry entry;
};

struct MemoryStats {
    size_t total_entries;
    size_t unique_types;
    float avg_importance;
    float memory_usage_mb;
    TimePoint oldest_entry;
    TimePoint newest_entry;
    std::unordered_map<std::string, size_t> type_distribution;
};

// Agent structures
struct Message {
    AgentID sender;
    AgentID receiver;
    std::string content;
    TimePoint timestamp;
    std::unordered_map<std::string, std::string> metadata;
};

struct Conversation {
    std::vector<Message> messages;
    TimePoint start_time;
    TimePoint last_updated;
    std::unordered_map<std::string, std::string> metadata;
};

struct ProcessOptions {
    size_t max_tokens = 2048;
    float temperature = 0.7f;
    bool use_memory = true;
    size_t memory_limit = 5;
    float memory_threshold = 0.5f;
    std::optional<std::string> context;
};

struct ProcessResult {
    std::string response;
    std::vector<MemoryID> relevant_memories;
    float confidence;
    std::chrono::milliseconds processing_time;
    std::unordered_map<std::string, std::string> metadata;
};

// Error types
class GloomError : public std::runtime_error {
public:
    explicit GloomError(const std::string& message) 
        : std::runtime_error(message) {}
};

class MemoryError : public GloomError {
public:
    explicit MemoryError(const std::string& message) 
        : GloomError(message) {}
};

class AgentError : public GloomError {
public:
    explicit AgentError(const std::string& message) 
        : GloomError(message) {}
};

// Callback types
using MemoryCallback = std::function<void(const MemoryEntry&)>;
using ProcessCallback = std::function<void(const ProcessResult&)>;
using ErrorCallback = std::function<void(const GloomError&)>;

// Plugin system
class Plugin {
public:
    virtual ~Plugin() = default;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual void initialize(Agent& agent) = 0;
};

using PluginPtr = std::shared_ptr<Plugin>;

// Event system
enum class EventType {
    MEMORY_STORED,
    MEMORY_RETRIEVED,
    MEMORY_CONSOLIDATED,
    AGENT_PROCESSING,
    AGENT_RESPONSE,
    ERROR
};

struct Event {
    EventType type;
    TimePoint timestamp;
    std::variant<
        MemoryEntry,
        ProcessResult,
        GloomError,
        std::string
    > data;
};

using EventCallback = std::function<void(const Event&)>;

// Utility types
struct VectorMetadata {
    size_t dimension;
    bool normalized;
    TimePoint created_at;
    std::unordered_map<std::string, std::string> metadata;
};

struct SearchParams {
    size_t limit = 10;
    float min_similarity = 0.5f;
    std::optional<std::string> type;
    std::optional<TimePoint> time_range_start;
    std::optional<TimePoint> time_range_end;
    bool include_metadata = true;
};

// Constants
constexpr size_t DEFAULT_EMBED_DIMENSION = 384;
constexpr float DEFAULT_TEMPERATURE = 0.7f;
constexpr size_t DEFAULT_MAX_TOKENS = 2048;
constexpr float DEFAULT_DECAY_RATE = 0.1f;
constexpr float DEFAULT_RETRIEVAL_THRESHOLD = 0.5f;

} // namespace gloom