#include "gloom/memory/semantic.hpp"
#include <algorithm>
#include <chrono>
#include <mutex>
#include <queue>
#include <unordered_set>

namespace gloom {
namespace memory {

class SemanticMemory::Impl {
public:
    struct Node {
        std::string id;
        std::string concept;
        std::unordered_map<std::string, std::string> attributes;
        std::vector<std::pair<std::string, double>> relationships;
        double importance;
        size_t access_count{0};
        std::chrono::system_clock::time_point created;
        std::chrono::system_clock::time_point last_accessed;
        
        Node(const std::string& node_id, const std::string& node_concept)
            : id(node_id)
            , concept(node_concept)
            , importance(0.0)
            , created(std::chrono::system_clock::now())
            , last_accessed(created) {}
    };
    
    std::unordered_map<std::string, Node> nodes;
    std::shared_mutex mutex;
    size_t capacity;
    
    explicit Impl(size_t max_capacity = 10000) : capacity(max_capacity) {}
};

SemanticMemory::SemanticMemory(size_t capacity)
    : pimpl(std::make_unique<Impl>(capacity)) {}

SemanticMemory::~SemanticMemory() = default;

std::string SemanticMemory::create_node(
    const std::string& concept,
    const std::unordered_map<std::string, std::string>& attributes
) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    if (pimpl->nodes.size() >= pimpl->capacity) {
        prune_nodes();
    }
    
    std::string id = generate_id();
    auto& node = pimpl->nodes.emplace(
        id, Impl::Node(id, concept)).first->second;
    
    node.attributes = attributes;
    return id;
}

bool SemanticMemory::add_relationship(
    const std::string& from_id,
    const std::string& to_id,
    double strength
) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    auto from_it = pimpl->nodes.find(from_id);
    auto to_it = pimpl->nodes.find(to_id);
    
    if (from_it == pimpl->nodes.end() || to_it == pimpl->nodes.end()) {
        return false;
    }
    
    // Update or add relationship
    auto& relationships = from_it->second.relationships;
    auto rel_it = std::find_if(relationships.begin(), relationships.end(),
        [&to_id](const auto& rel) { return rel.first == to_id; });
    
    if (rel_it != relationships.end()) {
        rel_it->second = strength;
    } else {
        relationships.emplace_back(to_id, strength);
    }
    
    return true;
}

std::optional<SemanticNode> SemanticMemory::get_node(const std::string& id) {
    std::shared_lock<std::shared_mutex> lock(pimpl->mutex);
    
    auto it = pimpl->nodes.find(id);
    if (it == pimpl->nodes.end()) {
        return std::nullopt;
    }
    
    auto& node = it->second;
    node.last_accessed = std::chrono::system_clock::now();
    node.access_count++;
    
    return convert_to_semantic_node(node);
}

std::vector<SemanticNode> SemanticMemory::search(
    const SemanticQuery& query,
    size_t limit
) {
    std::shared_lock<std::shared_mutex> lock(pimpl->mutex);
    std::vector<SemanticNode> results;
    
    for (const auto& [_, node] : pimpl->nodes) {
        if (matches_query(node, query)) {
            results.push_back(convert_to_semantic_node(node));
        }
    }
    
    // Sort by relevance
    std::sort(results.begin(), results.end(),
        [this, &query](const auto& a, const auto& b) {
            const auto& node_a = pimpl->nodes.at(a.id);
            const auto& node_b = pimpl->nodes.at(b.id);
            return calculate_relevance(node_a, query) >
                   calculate_relevance(node_b, query);
        });
    
    if (limit > 0 && results.size() > limit) {
        results.resize(limit);
    }
    
    // Update access metrics
    for (const auto& result : results) {
        auto it = pimpl->nodes.find(result.id);
        if (it != pimpl->nodes.end()) {
            it->second.last_accessed = std::chrono::system_clock::now();
            it->second.access_count++;
        }
    }
    
    return results;
}

std::vector<SemanticNode> SemanticMemory::get_related_nodes(
    const std::string& id,
    double min_strength,
    size_t limit
) {
    std::shared_lock<std::shared_mutex> lock(pimpl->mutex);
    std::vector<SemanticNode> related;
    
    auto it = pimpl->nodes.find(id);
    if (it == pimpl->nodes.end()) {
        return related;
    }
    
    for (const auto& [rel_id, strength] : it->second.relationships) {
        if (strength >= min_strength) {
            auto rel_it = pimpl->nodes.find(rel_id);
            if (rel_it != pimpl->nodes.end()) {
                related.push_back(convert_to_semantic_node(rel_it->second));
            }
        }
    }
    
    // Sort by relationship strength
    std::sort(related.begin(), related.end(),
        [&it](const auto& a, const auto& b) {
            auto find_strength = [&](const std::string& rel_id) {
                auto rel = std::find_if(
                    it->second.relationships.begin(),
                    it->second.relationships.end(),
                    [&rel_id](const auto& r) { return r.first == rel_id; }
                );
                return rel != it->second.relationships.end() ? rel->second : 0.0;
            };
            return find_strength(a.id) > find_strength(b.id);
        });
    
    if (limit > 0 && related.size() > limit) {
        related.resize(limit);
    }
    
    return related;
}

void SemanticMemory::update_node_importance(const std::string& id, double importance) {
    std::unique_lock<std::shared_mutex> lock(pimpl->mutex);
    
    auto it = pimpl->nodes.find(id);
    if (it != pimpl->nodes.end()) {
        it->second.importance = std::clamp(importance, 0.0, 1.0);
    }
}

void SemanticMemory::prune_nodes() {
    std::vector<std::pair<std::string, double>> scores;
    auto now = std::chrono::system_clock::now();
    
    for (const auto& [id, node] : pimpl->nodes) {
        double score = calculate_retention_score(node, now);
        scores.emplace_back(id, score);
    }
    
    std::sort(scores.begin(), scores.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    size_t to_remove = pimpl->nodes.size() - (pimpl->capacity * 0.9);
    for (size_t i = 0; i < to_remove && i < scores.size(); ++i) {
        pimpl->nodes.erase(scores[i].first);
    }
}

bool SemanticMemory::matches_query(const Impl::Node& node, const SemanticQuery& query) {
    // Concept match
    if (!query.concept.empty() && node.concept != query.concept) {
        return false;
    }
    
    // Attribute match
    for (const auto& [key, value] : query.attributes) {
        auto it = node.attributes.find(key);
        if (it == node.attributes.end() || it->second != value) {
            return false;
        }
    }
    
    return true;
}

double SemanticMemory::calculate_relevance(
    const Impl::Node& node,
    const SemanticQuery& query
) {
    auto now = std::chrono::system_clock::now();
    auto age = std::chrono::duration_cast<std::chrono::hours>(
        now - node.created).count();
    
    double recency_score = 1.0 / (1.0 + std::log1p(age));
    double access_score = std::log1p(node.access_count);
    double importance_score = node.importance;
    double relationship_score = std::log1p(node.relationships.size());
    
    return (recency_score * 0.2) +
           (access_score * 0.2) +
           (importance_score * 0.4) +
           (relationship_score * 0.2);
}

double SemanticMemory::calculate_retention_score(
    const Impl::Node& node,
    const std::chrono::system_clock::time_point& now
) {
    auto age = std::chrono::duration_cast<std::chrono::hours>(
        now - node.created).count();
    auto last_access = std::chrono::duration_cast<std::chrono::hours>(
        now - node.last_accessed).count();
    
    double age_score = 1.0 / (1.0 + std::log1p(age));
    double access_recency_score = 1.0 / (1.0 + std::log1p(last_access));
    double access_frequency_score = std::log1p(node.access_count);
    double importance_score = node.importance;
    double connectivity_score = std::log1p(node.relationships.size());
    
    return (age_score * 0.15) +
           (access_recency_score * 0.25) +
           (access_frequency_score * 0.2) +
           (importance_score * 0.25) +
           (connectivity_score * 0.15);
}

SemanticNode SemanticMemory::convert_to_semantic_node(const Impl::Node& node) {
    SemanticNode result;
    result.id = node.id;
    result.concept = node.concept;
    result.attributes = node.attributes;
    result.importance = node.importance;
    return result;
}

std::string SemanticMemory::generate_id() {
    static std::atomic<uint64_t> counter{0};
    return "sem_" + std::to_string(++counter);
}

} // namespace memory
} // namespace gloom