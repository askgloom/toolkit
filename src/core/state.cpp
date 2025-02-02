#include "state.hpp"
#include "../utils/logger.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>

namespace solana {
namespace toolkit {

namespace {
    static Logger logger("State");
}

State::State() 
    : status_(Status::INITIALIZING)
    , initialized_(false)
    , last_updated_(std::chrono::system_clock::now()) {
}

State::~State() {
    clear();
}

bool State::initialize() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    
    if (initialized_) {
        logger.warn("State already initialized");
        return true;
    }
    
    try {
        // Initialize containers
        connections_.clear();
        transactions_.clear();
        cache_.clear();
        
        status_ = Status::READY;
        initialized_ = true;
        last_updated_ = std::chrono::system_clock::now();
        
        logger.info("State initialized successfully");
        return true;
    } catch (const std::exception& e) {
        logger.error("Failed to initialize state: {}", e.what());
        status_ = Status::ERROR;
        return false;
    }
}

bool State::is_initialized() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return initialized_;
}

State::Status State::get_status() const {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_;
}

void State::set_status(Status status) {
    std::lock_guard<std::mutex> lock(status_mutex_);
    status_ = status;
    last_updated_ = std::chrono::system_clock::now();
    log_state_change("Status changed to: " + get_status_string());
}

std::string State::get_status_string() const {
    switch (status_) {
        case Status::INITIALIZING: return "INITIALIZING";
        case Status::READY:        return "READY";
        case Status::PROCESSING:   return "PROCESSING";
        case Status::ERROR:        return "ERROR";
        case Status::SHUTDOWN:     return "SHUTDOWN";
        default:                   return "UNKNOWN";
    }
}

std::string State::add_connection(const std::string& endpoint) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    // Generate unique connection ID
    static uint64_t next_id = 1;
    std::string id = "conn_" + std::to_string(next_id++);
    
    Connection conn{
        id,
        endpoint,
        std::chrono::system_clock::now(),
        true,
        {}
    };
    
    connections_[id] = conn;
    logger.debug("Added new connection: {} -> {}", id, endpoint);
    
    return id;
}

bool State::remove_connection(const std::string& id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    auto it = connections_.find(id);
    if (it == connections_.end()) {
        return false;
    }
    
    connections_.erase(it);
    logger.debug("Removed connection: {}", id);
    return true;
}

void State::update_connection_activity(const std::string& id) {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    auto it = connections_.find(id);
    if (it != connections_.end()) {
        it->second.last_active = std::chrono::system_clock::now();
        it->second.is_active = true;
    }
}

std::vector<State::Connection> State::get_active_connections() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    std::vector<Connection> active;
    for (const auto& [id, conn] : connections_) {
        if (conn.is_active) {
            active.push_back(conn);
        }
    }
    return active;
}

void State::track_transaction(const Transaction& transaction) {
    std::lock_guard<std::mutex> lock(transactions_mutex_);
    
    if (!validate_transaction(transaction)) {
        logger.warn("Invalid transaction rejected: {}", transaction.signature);
        return;
    }
    
    if (transactions_.size() >= MAX_TRANSACTIONS) {
        cleanup_old_transactions();
    }
    
    transactions_[transaction.signature] = transaction;
    logger.debug("Tracked new transaction: {}", transaction.signature);
}

std::optional<State::Transaction> State::get_transaction(
    const std::string& signature) const {
    std::lock_guard<std::mutex> lock(transactions_mutex_);
    
    auto it = transactions_.find(signature);
    if (it == transactions_.end()) {
        return std::nullopt;
    }
    return it->second;
}

void State::update_transaction_status(
    const std::string& signature,
    const std::string& status) {
    std::lock_guard<std::mutex> lock(transactions_mutex_);
    
    auto it = transactions_.find(signature);
    if (it != transactions_.end()) {
        it->second.status = status;
        it->second.timestamp = std::chrono::system_clock::now();
        logger.debug("Updated transaction status: {} -> {}", signature, status);
    }
}

std::vector<State::Transaction> State::get_recent_transactions(
    size_t limit) const {
    std::lock_guard<std::mutex> lock(transactions_mutex_);
    
    std::vector<Transaction> recent;
    recent.reserve(std::min(limit, transactions_.size()));
    
    for (const auto& [sig, tx] : transactions_) {
        recent.push_back(tx);
        if (recent.size() >= limit) break;
    }
    
    std::sort(recent.begin(), recent.end(),
        [](const Transaction& a, const Transaction& b) {
            return a.timestamp > b.timestamp;
        });
    
    return recent;
}

void State::cache_remove(const std::string& key) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.erase(key);
}

void State::cache_clear() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    cache_.clear();
}

bool State::update(const State& other) {
    try {
        {
            std::lock_guard<std::mutex> lock(status_mutex_);
            status_ = other.status_;
        }
        
        {
            std::lock_guard<std::mutex> lock(connections_mutex_);
            connections_ = other.connections_;
        }
        
        {
            std::lock_guard<std::mutex> lock(transactions_mutex_);
            transactions_ = other.transactions_;
        }
        
        last_updated_ = std::chrono::system_clock::now();
        return true;
    } catch (const std::exception& e) {
        logger.error("Failed to update state: {}", e.what());
        return false;
    }
}

void State::clear() {
    std::lock_guard<std::mutex> status_lock(status_mutex_);
    std::lock_guard<std::mutex> conn_lock(connections_mutex_);
    std::lock_guard<std::mutex> tx_lock(transactions_mutex_);
    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
    
    status_ = Status::SHUTDOWN;
    initialized_ = false;
    connections_.clear();
    transactions_.clear();
    cache_.clear();
    
    logger.info("State cleared");
}

size_t State::get_connection_count() const {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    return connections_.size();
}

size_t State::get_transaction_count() const {
    std::lock_guard<std::mutex> lock(transactions_mutex_);
    return transactions_.size();
}

size_t State::get_cache_size() const {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return cache_.size();
}

void State::cleanup_expired_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto now = std::chrono::system_clock::now();
    for (auto it = cache_.begin(); it != cache_.end();) {
        bool expired = std::visit([&](const auto& entry) {
            return !entry.is_valid || entry.expiry < now;
        }, it->second);
        
        if (expired) {
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
}

void State::cleanup_stale_connections() {
    std::lock_guard<std::mutex> lock(connections_mutex_);
    
    auto now = std::chrono::system_clock::now();
    for (auto it = connections_.begin(); it != connections_.end();) {
        if (now - it->second.last_active > CONNECTION_TIMEOUT) {
            logger.debug("Removing stale connection: {}", it->first);
            it = connections_.erase(it);
        } else {
            ++it;
        }
    }
}

void State::cleanup_old_transactions() {
    std::lock_guard<std::mutex> lock(transactions_mutex_);
    
    auto now = std::chrono::system_clock::now();
    for (auto it = transactions_.begin(); it != transactions_.end();) {
        if (now - it->second.timestamp > TRANSACTION_TTL) {
            it = transactions_.erase(it);
        } else {
            ++it;
        }
    }
}

void State::log_state_change(const std::string& message) {
    logger.info(message);
}

bool State::validate_transaction(const Transaction& transaction) const {
    return !transaction.signature.empty() && 
           !transaction.status.empty() && 
           transaction.slot > 0;
}

void State::prune_old_data() {
    cleanup_expired_cache();
    cleanup_stale_connections();
    cleanup_old_transactions();
}

} // namespace toolkit
} // namespace solana