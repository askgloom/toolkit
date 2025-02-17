#include <gloom/gloom.hpp>
#include <gloom/core/memory.hpp>
#include <gloom/utils/embeddings.hpp>
#include <spdlog/spdlog.h>
#include <iostream>
#include <deque>
#include <unordered_set>

using namespace gloom;

// Custom memory implementation with priority queue and categorization
class CustomMemory : public Memory {
public:
    explicit CustomMemory(const MemoryConfig& config)
        : Memory(config)
        , max_priority_entries_(100)
        , category_limit_(50) {
        spdlog::info("Initializing CustomMemory with {} categories limit", category_limit_);
    }

    // Override store to implement priority queue and categorization
    bool store(const MemoryEntry& entry) override {
        try {
            // Generate category for the entry
            std::string category = categorize_entry(entry);
            
            // Store in base memory
            if (!Memory::store(entry)) {
                return false;
            }

            // Update priority queue
            update_priority_queue(entry);

            // Update category index
            update_category_index(category, entry);

            spdlog::debug("Stored entry in category: {}", category);
            return true;
        } catch (const std::exception& e) {
            spdlog::error("Error storing memory: {}", e.what());
            return false;
        }
    }

    // Override retrieve to use priority and category-based search
    std::vector<MemorySearchResult> retrieve(
        const std::string& query,
        const SearchParams& params
    ) override {
        try {
            // Get category for query
            std::string query_category = categorize_query(query);
            
            // Get results from base memory
            auto base_results = Memory::retrieve(query, params);
            
            // Get results from priority queue
            auto priority_results = search_priority_queue(query, params);
            
            // Get results from category
            auto category_results = search_category(query_category, params);

            // Merge and rank results
            return merge_and_rank_results(
                base_results,
                priority_results,
                category_results,
                params.limit
            );
        } catch (const std::exception& e) {
            spdlog::error("Error retrieving memories: {}", e.what());
            return {};
        }
    }

    // Custom memory consolidation
    void consolidate() override {
        try {
            spdlog::info("Starting custom memory consolidation");

            // Consolidate base memory
            Memory::consolidate();

            // Consolidate priority queue
            consolidate_priority_queue();

            // Consolidate categories
            consolidate_categories();

            spdlog::info("Custom memory consolidation completed");
        } catch (const std::exception& e) {
            spdlog::error("Error during consolidation: {}", e.what());
        }
    }

    // Get custom memory statistics
    MemoryStats get_stats() const override {
        auto base_stats = Memory::get_stats();
        base_stats.metadata["priority_queue_size"] = 
            std::to_string(priority_queue_.size());
        base_stats.metadata["category_count"] = 
            std::to_string(category_index_.size());
        return base_stats;
    }

private:
    std::deque<MemoryEntry> priority_queue_;
    std::unordered_map<std::string, std::vector<MemoryEntry>> category_index_;
    const size_t max_priority_entries_;
    const size_t category_limit_;
    utils::EmbeddingGenerator embedding_generator_{384};

    // Helper methods
    std::string categorize_entry(const MemoryEntry& entry) {
        // Simple categorization based on content analysis
        // In practice, you might want to use more sophisticated methods
        if (entry.content.find("error") != std::string::npos) return "error";
        if (entry.content.find("warning") != std::string::npos) return "warning";
        if (entry.content.find("query") != std::string::npos) return "query";
        return "general";
    }

    std::string categorize_query(const std::string& query) {
        return categorize_entry(MemoryEntry{
            .content = query,
            .importance = 1.0f
        });
    }

    void update_priority_queue(const MemoryEntry& entry) {
        // Add to priority queue
        priority_queue_.push_back(entry);
        
        // Sort by importance
        std::sort(priority_queue_.begin(), priority_queue_.end(),
            [](const auto& a, const auto& b) {
                return a.importance > b.importance;
            });
        
        // Maintain size limit
        if (priority_queue_.size() > max_priority_entries_) {
            priority_queue_.pop_back();
        }
    }

    void update_category_index(const std::string& category, 
                             const MemoryEntry& entry) {
        // Add to category
        auto& category_entries = category_index_[category];
        category_entries.push_back(entry);
        
        // Sort category by timestamp
        std::sort(category_entries.begin(), category_entries.end(),
            [](const auto& a, const auto& b) {
                return a.timestamp > b.timestamp;
            });
        
        // Maintain category size limit
        if (category_entries.size() > category_limit_) {
            category_entries.pop_back();
        }
    }

    std::vector<MemorySearchResult> search_priority_queue(
        const std::string& query,
        const SearchParams& params
    ) {
        std::vector<MemorySearchResult> results;
        auto query_embedding = embedding_generator_.generate(query);

        for (const auto& entry : priority_queue_) {
            float similarity = calculate_similarity(
                query_embedding, entry.embedding
            );
            
            if (similarity >= params.min_similarity) {
                results.push_back({
                    .id = entry.id,
                    .similarity = similarity,
                    .entry = entry
                });
            }
        }

        return results;
    }

    std::vector<MemorySearchResult> search_category(
        const std::string& category,
        const SearchParams& params
    ) {
        std::vector<MemorySearchResult> results;
        auto it = category_index_.find(category);
        
        if (it != category_index_.end()) {
            for (const auto& entry : it->second) {
                results.push_back({
                    .id = entry.id,
                    .similarity = 1.0f,  // Category match
                    .entry = entry
                });
            }
        }

        return results;
    }

    std::vector<MemorySearchResult> merge_and_rank_results(
        const std::vector<MemorySearchResult>& base_results,
        const std::vector<MemorySearchResult>& priority_results,
        const std::vector<MemorySearchResult>& category_results,
        size_t limit
    ) {
        // Combine all results
        std::vector<MemorySearchResult> merged;
        merged.insert(merged.end(), base_results.begin(), base_results.end());
        merged.insert(merged.end(), priority_results.begin(), priority_results.end());
        merged.insert(merged.end(), category_results.begin(), category_results.end());

        // Remove duplicates
        std::unordered_set<MemoryID> seen;
        std::vector<MemorySearchResult> unique;
        
        for (const auto& result : merged) {
            if (seen.insert(result.id).second) {
                unique.push_back(result);
            }
        }

        // Sort by combined score (similarity * importance)
        std::sort(unique.begin(), unique.end(),
            [](const auto& a, const auto& b) {
                return (a.similarity * a.entry.importance) >
                       (b.similarity * b.entry.importance);
            });

        // Return top results
        return std::vector<MemorySearchResult>(
            unique.begin(),
            unique.begin() + std::min(unique.size(), limit)
        );
    }

    void consolidate_priority_queue() {
        // Remove old entries
        auto now = std::chrono::system_clock::now();
        priority_queue_.erase(
            std::remove_if(priority_queue_.begin(), priority_queue_.end(),
                [&](const auto& entry) {
                    auto age = std::chrono::duration_cast<std::chrono::hours>(
                        now - entry.timestamp
                    ).count();
                    return age > 24;  // Remove entries older than 24 hours
                }
            ),
            priority_queue_.end()
        );
    }

    void consolidate_categories() {
        // Remove empty categories
        for (auto it = category_index_.begin(); it != category_index_.end();) {
            if (it->second.empty()) {
                it = category_index_.erase(it);
            } else {
                ++it;
            }
        }
    }
};

// Example usage
int main() {
    try {
        // Configure logging
        spdlog::set_level(spdlog::level::debug);
        spdlog::info("Starting custom memory example");

        // Initialize custom memory
        MemoryConfig config{
            .capacity = 1000,
            .decay_rate = 0.1f,
            .retrieval_threshold = 0.5f,
            .embed_dimension = 384
        };

        CustomMemory memory(config);

        // Store some test memories
        std::vector<MemoryEntry> test_entries = {
            {
                .content = "Error: Connection failed",
                .importance = 0.9f,
                .timestamp = std::chrono::system_clock::now()
            },
            {
                .content = "Warning: Low memory",
                .importance = 0.7f,
                .timestamp = std::chrono::system_clock::now()
            },
            {
                .content = "Query: What is the status?",
                .importance = 0.5f,
                .timestamp = std::chrono::system_clock::now()
            }
        };

        for (const auto& entry : test_entries) {
            memory.store(entry);
        }

        // Retrieve memories
        SearchParams params{
            .limit = 5,
            .min_similarity = 0.5f
        };

        auto results = memory.retrieve("error", params);

        // Print results
        spdlog::info("Retrieved {} results", results.size());
        for (const auto& result : results) {
            spdlog::info("Result: {} (similarity: {:.2f})",
                result.entry.content, result.similarity);
        }

        // Print statistics
        auto stats = memory.get_stats();
        spdlog::info("Memory stats:");
        spdlog::info("- Total entries: {}", stats.total_entries);
        spdlog::info("- Priority queue size: {}", 
            stats.metadata["priority_queue_size"]);
        spdlog::info("- Category count: {}", 
            stats.metadata["category_count"]);

        return 0;
    } catch (const std::exception& e) {
        spdlog::error("Error in main: {}", e.what());
        return 1;
    }
}