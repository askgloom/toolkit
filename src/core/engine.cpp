#include "engine.hpp"
#include <spdlog/spdlog.h>
#include <chrono>
#include <thread>

namespace solana {
namespace toolkit {

constexpr char Engine::VERSION[];

Engine::Engine(const Config& config) 
    : config_(config),
      state_(std::make_shared<State>()),
      logger_(std::make_unique<Logger>(config.log_level)) {
    
    logger_->info("Initializing Solana toolkit engine v{}", VERSION);
}

Engine::~Engine() {
    if (running_) {
        stop();
    }
}

bool Engine::initialize() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    try {
        // Initialize Solana client
        client_ = std::make_shared<solana::Client>(
            config_.rpc_url,
            config_.ws_url,
            config_.timeout
        );
        
        // Initialize state
        if (!state_->initialize()) {
            logger_->error("Failed to initialize state");
            return false;
        }
        
        logger_->info("Engine initialized successfully");
        return true;
    } catch (const std::exception& e) {
        logger_->error("Initialization error: {}", e.what());
        return false;
    }
}

bool Engine::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (running_) {
        logger_->warn("Engine already running");
        return true;
    }
    
    try {
        running_ = true;
        
        // Start worker thread
        worker_ = std::make_unique<std::thread>([this]() {
            process_queue();
        });
        
        // Start periodic cleanup
        std::thread([this]() {
            while (running_) {
                std::this_thread::sleep_for(CLEANUP_INTERVAL);
                cleanup_subscriptions();
            }
        }).detach();
        
        logger_->info("Engine started successfully");
        return true;
    } catch (const std::exception& e) {
        running_ = false;
        logger_->error("Failed to start engine: {}", e.what());
        return false;
    }
}

void Engine::stop() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!running_) {
        return;
    }
    
    logger_->info("Stopping engine...");
    
    running_ = false;
    cv_.notify_all();
    
    if (worker_ && worker_->joinable()) {
        worker_->join();
    }
    
    // Cleanup subscriptions
    {
        std::lock_guard<std::mutex> sub_lock(subscriptions_mutex_);
        subscriptions_.clear();
    }
    
    // Clear request queue
    {
        std::lock_guard<std::mutex> queue_lock(queue_mutex_);
        std::queue<std::function<void()>> empty;
        std::swap(request_queue_, empty);
    }
    
    logger_->info("Engine stopped successfully");
}

bool Engine::is_running() const {
    return running_;
}

std::shared_ptr<solana::Client> Engine::get_client() {
    std::lock_guard<std::mutex> lock(mutex_);
    return client_;
}

bool Engine::set_client(std::shared_ptr<solana::Client> client) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!client) {
        return false;
    }
    client_ = client;
    return true;
}

std::shared_ptr<State> Engine::get_state() {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

bool Engine::update_state(const State& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_->update(state);
}

SubscriptionId Engine::subscribe(
    const std::string& event,
    std::function<void(const Event&)> callback
) {
    if (!callback) {
        logger_->error("Invalid callback provided for subscription");
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    static SubscriptionId next_id = 1;
    
    SubscriptionId id = next_id++;
    subscriptions_[id] = callback;
    
    logger_->debug("Created subscription {} for event {}", id, event);
    return id;
}

bool Engine::unsubscribe(SubscriptionId id) {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    
    auto it = subscriptions_.find(id);
    if (it == subscriptions_.end()) {
        return false;
    }
    
    subscriptions_.erase(it);
    logger_->debug("Removed subscription {}", id);
    return true;
}

Engine::Metrics Engine::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void Engine::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = Metrics{};
}

const Engine::Config& Engine::get_config() const {
    return config_;
}

bool Engine::update_config(const Config& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    logger_->set_level(config.log_level);
    return true;
}

std::string Engine::get_version() const {
    return VERSION;
}

std::string Engine::get_status() const {
    return running_ ? "running" : "stopped";
}

void Engine::process_queue() {
    while (running_) {
        std::function<void()> task;
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]() {
                return !request_queue_.empty() || !running_;
            });
            
            if (!running_) {
                break;
            }
            
            task = std::move(request_queue_.front());
            request_queue_.pop();
        }
        
        try {
            task();
            metrics_.requests_processed++;
        } catch (const std::exception& e) {
            handle_error(e.what());
        }
    }
}

void Engine::update_metrics(const std::chrono::milliseconds& latency) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.last_request = std::chrono::system_clock::now();
    
    // Update average latency
    double current_avg = metrics_.average_latency;
    uint64_t count = metrics_.requests_processed;
    
    metrics_.average_latency = 
        (current_avg * count + latency.count()) / (count + 1);
}

void Engine::cleanup_subscriptions() {
    std::lock_guard<std::mutex> lock(subscriptions_mutex_);
    
    // Implement subscription cleanup logic here
    // For example, remove expired subscriptions
    logger_->debug("Performing subscription cleanup");
}

void Engine::handle_error(const std::string& error) {
    logger_->error("Engine error: {}", error);
    metrics_.errors_count++;
}

} // namespace toolkit
} // namespace solana