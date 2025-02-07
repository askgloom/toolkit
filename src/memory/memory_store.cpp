#include "gloom/memory/memory_store.hpp"
#include <algorithm>
#include <chrono>
#include <mutex>
#include <shared_mutex>

namespace gloom {
namespace memory {

class MemoryStore::Impl {
public:
    std::unordered_map<std::string, Memory> memories;
    std::shared_mutex mutex;
    size_t capacity;
    
    explicit Impl(size_t max_capacity = 1000) : capacity(max_capacity) {}
};

MemoryStore::MemoryStore(size_t capacity) : pimpl(std::make_unique<Impl>(capacity)) {}
MemoryStore::~MemoryStore() = default;

bool MemoryStore::store(const Memory& memory) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    if (pimpl->memories.size() >= pimpl->capacity) {
        prune();
    }
    
    auto timestamp = std::chrono::system_clock::now();
    Memory mem = memory;
    mem.timestamp = timestamp;
    mem.last_accessed = timestamp;
    
    pimpl->memories[mem.id] = std::move(mem);
    return true;
}

std::optional<Memory> MemoryStore::retrieve(const std::string& id) {
    std::shared_lock<std::shared_mutex> lock(pimpl->mutex);
    
    auto it = pimpl->memories.find(id);
    if (it != pimpl->memories.end()) {
        it->second.last_accessed = std::chrono::system_clock::now();
        it->second.access_count++;
        return it->second;
    }
    
    return std::nullopt;
}

std::vector<Memory> MemoryStore::search(const Query& query, size_t limit) {
    std::shared_lock<std::shared_mutex> lock(pimpl->mutex);
    std::vector<Memory> results;
    
    for (const auto& [_, memory] : pimpl->memories) {
        if (matches_query(memory, query)) {
            results.push_back(memory);
        }
    }
    
    // Sort by relevance
    std::sort(results.begin(), results.end(), 
        [](const Memory& a, const Memory& b) {
            return calculate_relevance(a) > calculate_relevance(b);
        });
    
    // Limit results
    if (limit > 0 && results.size() > limit) {
        results.resize(limit);
    }
    
    // Update access metrics
    for (const auto& result : results) {
        auto it = pimpl->memories.find(result.id);
        if (it != pimpl->memories.end()) {
            it->second.last_accessed = std::chrono::system_clock::now();
            it->second.access_count++;
        }
    }
    
    return results;
}

bool MemoryStore::update(const std::string& id, const MemoryUpdate& update) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    auto it = pimpl->memories.find(id);
    if (it == pimpl->memories.end()) {
        return false;
    }
    
    // Apply updates
    if (update.content) it->second.content = *update.content;
    if (update.tags) it->second.tags = *update.tags;
    if (update.metadata) it->second.metadata = *update.metadata;
    
    it->second.last_modified = std::chrono::system_clock::now();
    return true;
}

bool MemoryStore::remove(const std::string& id) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    return pimpl->memories.erase(id) > 0;
}

void MemoryStore::clear() {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    pimpl->memories.clear();
}

size_t MemoryStore::size() const {
    std::shared_lock<std::shared_mutex> lock(pimpl->mutex);
    return pimpl->memories.size();
}

void MemoryStore::prune() {
    if (pimpl->memories.size() < pimpl->capacity) {
        return;
    }
    
    std::vector<std::pair<std::string, double>> scores;
    auto now = std::chrono::system_clock::now();
    
    // Calculate retention scores
    for (const auto& [id, memory] : pimpl->memories) {
        double score = calculate_retention_score(memory, now);
        scores.emplace_back(id, score);
    }
    
    // Sort by score ascending (lower scores will be removed first)
    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    // Remove lowest scoring memories until we're under capacity
    size_t to_remove = pimpl->memories.size() - (pimpl->capacity * 0.9);
    for (size_t i = 0; i < to_remove && i < scores.size(); ++i) {
        pimpl->memories.erase(scores[i].first);
    }
}

bool MemoryStore::matches_query(const Memory& memory, const Query& query) {
    // Content match
    if (!query.content.empty() && 
        memory.content.find(query.content) == std::string::npos) {
        return false;
    }
    
    // Tag match
    if (!query.tags.empty()) {
        bool has_tag = false;
        for (const auto& tag : query.tags) {
            if (memory.tags.find(tag) != memory.tags.end()) {
                has_tag = true;
                break;
            }
        }
        if (!has_tag) return false;
    }
    
    // Timestamp range match
    if (query.start_time && memory.timestamp < *query.start_time) return false;
    if (query.end_time && memory.timestamp > *query.end_time) return false;
    
    return true;
}

double MemoryStore::calculate_relevance(const Memory& memory) {
    auto now = std::chrono::system_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::hours>(
        now - memory.timestamp).count();
    
    // Factors in relevance calculation:
    // 1. Recency (newer memories are more relevant)
    // 2. Access frequency (more accessed memories are more relevant)
    // 3. Importance (explicitly marked important memories are more relevant)
    
    double recency_score = 1.0 / (1.0 + std::log1p(age));
    double access_score = std::log1p(memory.access_count);
    double importance_score = memory.importance;
    
    return (recency_score * 0.4) + 
           (access_score * 0.3) + 
           (importance_score * 0.3);
}

double MemoryStore::calculate_retention_score(
    const Memory& memory, 
    const std::chrono::system_clock::time_point& now
) {
    auto age = std::chrono::duration_cast<std::chrono::hours>(
        now - memory.timestamp).count();
    auto last_access = std::chrono::duration_cast<std::chrono::hours>(
        now - memory.last_accessed).count();
    
    // Factors in retention calculation:
    // 1. Age (older memories are more likely to be pruned)
    // 2. Access recency (recently accessed memories are retained)
    // 3. Access frequency (frequently accessed memories are retained)
    // 4. Importance (important memories are retained)
    
    double age_score = 1.0 / (1.0 + std::log1p(age));
    double access_recency_score = 1.0 / (1.0 + std::log1p(last_access));
    double access_frequency_score = std::log1p(memory.access_count);
    double importance_score = memory.importance;
    
    return (age_score * 0.2) + 
           (access_recency_score * 0.3) + 
           (access_frequency_score * 0.2) + 
           (importance_score * 0.3);
}

} // namespace memory
} // namespace gloom