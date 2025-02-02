#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <atomic>
#include <chrono>

#include "state.hpp"
#include "../solana/client.hpp"
#include "../utils/logger.hpp"
#include "../utils/config.hpp"

namespace solana {
namespace toolkit {

class Engine {
public:
    struct Config {
        std::string rpc_url;
        std::string ws_url;
        uint32_t max_connections{10};
        std::chrono::milliseconds timeout{5000};
        bool enable_metrics{true};
        std::string log_level{"info"};
    };

    struct Metrics {
        std::atomic<uint64_t> requests_processed{0};
        std::atomic<uint64_t> errors_count{0};
        std::chrono::system_clock::time_point last_request;
        double average_latency{0.0};
    };

    // Constructor
    explicit Engine(const Config& config);
    ~Engine();

    // Delete copy constructor and assignment
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Core functionality
    bool initialize();
    bool start();
    void stop();
    bool is_running() const;

    // Client management
    std::shared_ptr<solana::Client> get_client();
    bool set_client(std::shared_ptr<solana::Client> client);

    // State management
    std::shared_ptr<State> get_state();
    bool update_state(const State& state);

    // Request handling
    template<typename T>
    Result<T> process_request(const Request<T>& request);

    // Subscription management
    SubscriptionId subscribe(const std::string& event, 
                           std::function<void(const Event&)> callback);
    bool unsubscribe(SubscriptionId id);

    // Metrics
    Metrics get_metrics() const;
    void reset_metrics();

    // Configuration
    const Config& get_config() const;
    bool update_config(const Config& config);

    // Utility functions
    std::string get_version() const;
    std::string get_status() const;

private:
    // Internal types
    using SubscriptionMap = std::unordered_map<SubscriptionId, 
                                             std::function<void(const Event&)>>;
    using RequestQueue = std::queue<std::function<void()>>;

    // Internal methods
    void process_queue();
    void update_metrics(const std::chrono::milliseconds& latency);
    void cleanup_subscriptions();
    void handle_error(const std::string& error);

    // Member variables
    Config config_;
    std::shared_ptr<State> state_;
    std::shared_ptr<solana::Client> client_;
    std::unique_ptr<Logger> logger_;
    
    std::atomic<bool> running_{false};
    std::mutex mutex_;
    
    SubscriptionMap subscriptions_;
    std::mutex subscriptions_mutex_;
    
    RequestQueue request_queue_;
    std::mutex queue_mutex_;
    
    Metrics metrics_;
    std::mutex metrics_mutex_;

    // Worker thread
    std::unique_ptr<std::thread> worker_;
    std::condition_variable cv_;

    // Constants
    static constexpr char VERSION[] = "0.1.0";
    static constexpr size_t MAX_QUEUE_SIZE = 1000;
    static constexpr auto CLEANUP_INTERVAL = std::chrono::minutes(5);
};

// Template implementations
template<typename T>
Result<T> Engine::process_request(const Request<T>& request) {
    if (!running_) {
        return Result<T>::error("Engine not running");
    }

    auto start_time = std::chrono::steady_clock::now();
    
    try {
        auto result = request.execute(client_);
        
        auto end_time = std::chrono::steady_clock::now();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>
                      (end_time - start_time);
        
        update_metrics(latency);
        
        return result;
    } catch (const std::exception& e) {
        handle_error(e.what());
        return Result<T>::error(e.what());
    }
}

} // namespace toolkit
} // namespace solana