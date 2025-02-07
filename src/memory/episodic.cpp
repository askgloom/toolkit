#include "gloom/memory/episodic.hpp"
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>

namespace gloom {
namespace memory {

class EpisodicMemory::Impl {
public:
    struct Episode {
        std::string id;
        std::chrono::system_clock::time_point timestamp;
        std::vector<Memory> memories;
        std::unordered_map<std::string, std::string> context;
        double importance;
        size_t access_count{0};
        std::chrono::system_clock::time_point last_accessed;
        
        Episode(const std::string& episode_id) 
            : id(episode_id)
            , timestamp(std::chrono::system_clock::now())
            , last_accessed(timestamp)
            , importance(0.0) {}
    };
    
    std::unordered_map<std::string, Episode> episodes;
    std::shared_mutex mutex;
    size_t capacity;
    size_t max_memories_per_episode;
    
    explicit Impl(size_t max_episodes = 100, size_t max_memories = 50) 
        : capacity(max_episodes)
        , max_memories_per_episode(max_memories) {}
};

EpisodicMemory::EpisodicMemory(size_t max_episodes, size_t max_memories_per_episode)
    : pimpl(std::make_unique<Impl>(max_episodes, max_memories_per_episode)) {}

EpisodicMemory::~EpisodicMemory() = default;

std::string EpisodicMemory::create_episode(const std::unordered_map<std::string, std::string>& context) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    if (pimpl->episodes.size() >= pimpl->capacity) {
        prune_episodes();
    }
    
    std::string id = generate_id();
    auto& episode = pimpl->episodes.emplace(
        id, Impl::Episode(id)).first->second;
    
    episode.context = context;
    return id;
}

bool EpisodicMemory::add_memory(const std::string& episode_id, const Memory& memory) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    auto it = pimpl->episodes.find(episode_id);
    if (it == pimpl->episodes.end()) {
        return false;
    }
    
    auto& episode = it->second;
    if (episode.memories.size() >= pimpl->max_memories_per_episode) {
        prune_memories(episode);
    }
    
    episode.memories.push_back(memory);
    update_episode_importance(episode);
    return true;
}

std::optional<std::vector<Memory>> EpisodicMemory::recall_episode(const std::string& episode_id) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    auto it = pimpl->episodes.find(episode_id);
    if (it == pimpl->episodes.end()) {
        return std::nullopt;
    }
    
    auto& episode = it->second;
    episode.last_accessed = std::chrono::system_clock::now();
    episode.access_count++;
    
    return episode.memories;
}

std::vector<std::pair<std::string, std::vector<Memory>>> 
EpisodicMemory::search(const EpisodeQuery& query, size_t limit) {
    std::shared_lock<std::shared_mutex> lock(pimpl->mutex);
    std::vector<std::pair<std::string, std::vector<Memory>>> results;
    
    for (const auto& [id, episode] : pimpl->episodes) {
        if (matches_query(episode, query)) {
            results.emplace_back(id, episode.memories);
        }
    }
    
    // Sort by episode importance and recency
    std::sort(results.begin(), results.end(),
        [this, &query](const auto& a, const auto& b) {
            const auto& episode_a = pimpl->episodes.at(a.first);
            const auto& episode_b = pimpl->episodes.at(b.first);
            return calculate_relevance(episode_a, query) > 
                   calculate_relevance(episode_b, query);
        });
    
    if (limit > 0 && results.size() > limit) {
        results.resize(limit);
    }
    
    return results;
}

void EpisodicMemory::prune_episodes() {
    std::vector<std::pair<std::string, double>> scores;
    auto now = std::chrono::system_clock::now();
    
    for (const auto& [id, episode] : pimpl->episodes) {
        double score = calculate_retention_score(episode, now);
        scores.emplace_back(id, score);
    }
    
    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    size_t to_remove = pimpl->episodes.size() - (pimpl->capacity * 0.9);
    for (size_t i = 0; i < to_remove && i < scores.size(); ++i) {
        pimpl->episodes.erase(scores[i].first);
    }
}

void EpisodicMemory::prune_memories(Impl::Episode& episode) {
    auto& memories = episode.memories;
    std::vector<std::pair<size_t, double>> scores;
    
    for (size_t i = 0; i < memories.size(); ++i) {
        scores.emplace_back(i, calculate_memory_importance(memories[i]));
    }
    
    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    size_t to_remove = memories.size() - (pimpl->max_memories_per_episode * 0.9);
    std::vector<Memory> retained_memories;
    retained_memories.reserve(memories.size() - to_remove);
    
    std::unordered_set<size_t> indices_to_remove;
    for (size_t i = 0; i < to_remove && i < scores.size(); ++i) {
        indices_to_remove.insert(scores[i].first);
    }
    
    for (size_t i = 0; i < memories.size(); ++i) {
        if (indices_to_remove.find(i) == indices_to_remove.end()) {
            retained_memories.push_back(std::move(memories[i]));
        }
    }
    
    memories = std::move(retained_memories);
}

void EpisodicMemory::update_episode_importance(Impl::Episode& episode) {
    double total_importance = 0.0;
    for (const auto& memory : episode.memories) {
        total_importance += calculate_memory_importance(memory);
    }
    
    episode.importance = total_importance / 
        (episode.memories.empty() ? 1.0 : episode.memories.size());
}

bool EpisodicMemory::matches_query(const Impl::Episode& episode, const EpisodeQuery& query) {
    // Time range check
    if (query.start_time && episode.timestamp < *query.start_time) return false;
    if (query.end_time && episode.timestamp > *query.end_time) return false;
    
    // Context match
    for (const auto& [key, value] : query.context) {
        auto it = episode.context.find(key);
        if (it == episode.context.end() || it->second != value) {
            return false;
        }
    }
    
    // Content match
    if (!query.content.empty()) {
        bool content_match = false;
        for (const auto& memory : episode.memories) {
            if (memory.content.find(query.content) != std::string::npos) {
                content_match = true;
                break;
            }
        }
        if (!content_match) return false;
    }
    
    return true;
}

double EpisodicMemory::calculate_relevance(
    const Impl::Episode& episode,
    const EpisodeQuery& query
) {
    auto now = std::chrono::system_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::hours>(
        now - episode.timestamp).count();
    
    double recency_score = 1.0 / (1.0 + std::log1p(age));
    double access_score = std::log1p(episode.access_count);
    double importance_score = episode.importance;
    
    return (recency_score * 0.3) + 
           (access_score * 0.3) + 
           (importance_score * 0.4);
}

double EpisodicMemory::calculate_retention_score(
    const Impl::Episode& episode,
    const std::chrono::system_clock::time_point& now
) {
    auto age = std::chrono::duration_cast<std::chrono::hours>(
        now - episode.timestamp).count();
    auto last_access = std::chrono::duration_cast<std::chrono::hours>(
        now - episode.last_accessed).count();
    
    double age_score = 1.0 / (1.0 + std::log1p(age));
    double access_recency_score = 1.0 / (1.0 + std::log1p(last_access));
    double access_frequency_score = std::log1p(episode.access_count);
    double importance_score = episode.importance;
    
    return (age_score * 0.2) + 
           (access_recency_score * 0.3) + 
           (access_frequency_score * 0.2) + 
           (importance_score * 0.3);
}

double EpisodicMemory::calculate_memory_importance(const Memory& memory) {
    return memory.importance * (1.0 + std::log1p(memory.access_count));
}

std::string EpisodicMemory::generate_id() {
    static std::atomic<uint64_t> counter{0};
    return "ep_" + std::to_string(++counter);
}

} // namespace memory
} // namespace gloom