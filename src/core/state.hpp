#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <chrono>
#include <optional>
#include <variant>

namespace solana {
namespace toolkit {

class State {
public:
    // Status tracking
    enum class Status {
        INITIALIZING,
        READY,
        PROCESSING,
        ERROR,
        SHUTDOWN
    };

    // Connection state
    struct Connection {
        std::string id;
        std::string endpoint;
        std::chrono::system_clock::time_point last_active;
        bool is_active;
        std::unordered_map<std::string, std::string> metadata;
    };

    // Transaction state
    struct Transaction {
        std::string signature;
        std::string status;
        std::chrono::system_clock::time_point timestamp;
        uint64_t slot;
        std::optional<std::string> error;
        std::unordered_map<std::string, std::variant<
            std::string,
            int64_t,
            double,
            bool
        >> data;
    };

    // Cache entry
    template<typename T>
    struct CacheEntry {
        T value;
        std::chrono::system_clock::time_point expiry;
        bool is_valid;
    };

    // Constructor and destructor
    State();
    ~State();

    // Delete copy constructor and assignment
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    // Initialization
    bool initialize();
    bool is_initialized() const;

    // Status management
    Status get_status() const;
    void set_status(Status status);
    std::string get_status_string() const;

    // Connection management
    std::string add_connection(const std::string& endpoint);
    bool remove_connection(const std::string& id);
    void update_connection_activity(const std::string& id);
    std::vector<Connection> get_active_connections() const;

    // Transaction management
    void track_transaction(const Transaction& transaction);
    std::optional<Transaction> get_transaction(const std::string& signature) const;
    void update_transaction_status(const std::string& signature, 
                                 const std::string& status);
    std::vector<Transaction> get_recent_transactions(size_t limit = 100) const;

    // Cache management
    template<typename T>
    void cache_set(const std::string& key, 
                  const T& value, 
                  std::chrono::seconds ttl);

    template<typename T>
    std::optional<T> cache_get(const std::string& key) const;

    void cache_remove(const std::string& key);
    void cache_clear();

    // State updates
    bool update(const State& other);
    void clear();

    // Metrics
    size_t get_connection_count() const;
    size_t get_transaction_count() const;
    size_t get_cache_size() const;

    // Cleanup
    void cleanup_expired_cache();
    void cleanup_stale_connections();
    void cleanup_old_transactions();

private:
    // Internal types
    using Cache = std::unordered_map<std::string, 
                                   std::variant<
                                       CacheEntry<std::string>,
                                       CacheEntry<int64_t>,
                                       CacheEntry<double>,
                                       CacheEntry<bool>
                                   >>;

    // Internal methods
    void log_state_change(const std::string& message);
    bool validate_transaction(const Transaction& transaction) const;
    void prune_old_data();

    // Member variables
    Status status_;
    bool initialized_;
    std::chrono::system_clock::time_point last_updated_;
    
    std::unordered_map<std::string, Connection> connections_;
    std::unordered_map<std::string, Transaction> transactions_;
    Cache cache_;

    // Mutexes for thread safety
    mutable std::mutex status_mutex_;
    mutable std::mutex connections_mutex_;
    mutable std::mutex transactions_mutex_;
    mutable std::mutex cache_mutex_;

    // Constants
    static constexpr size_t MAX_TRANSACTIONS = 10000;
    static constexpr size_t MAX_CACHE_SIZE = 1000;
    static constexpr auto CONNECTION_TIMEOUT = std::chrono::minutes(5);
    static constexpr auto TRANSACTION_TTL = std::chrono::hours(24);
};

// Template implementations
template<typename T>
void State::cache_set(const std::string& key, 
                     const T& value, 
                     std::chrono::seconds ttl) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto expiry = std::chrono::system_clock::now() + ttl;
    CacheEntry<T> entry{value, expiry, true};
    
    if (cache_.size() >= MAX_CACHE_SIZE) {
        cleanup_expired_cache();
        if (cache_.size() >= MAX_CACHE_SIZE) {
            // Remove oldest entry if still at capacity
            cache_.erase(cache_.begin());
        }
    }
    
    cache_[key] = entry;
}

template<typename T>
std::optional<T> State::cache_get(const std::string& key) const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = cache_.find(key);
    if (it == cache_.end()) {
        return std::nullopt;
    }
    
    const auto& entry = std::get<CacheEntry<T>>(it->second);
    if (!entry.is_valid || 
        std::chrono::system_clock::now() > entry.expiry) {
        return std::nullopt;
    }
    
    return entry.value;
}

} // namespace toolkit
} // namespace solana