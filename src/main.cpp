#include <gloom/gloom.hpp>
#include <gloom/core/agent.hpp>
#include <gloom/core/memory.hpp>
#include <gloom/utils/embeddings.hpp>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include <iostream>
#include <string>
#include <memory>

using namespace gloom;

// Command line interface setup
void setup_cli(CLI::App& app, AgentConfig& config, std::string& input) {
    app.add_option("-n,--name", config.name, "Agent name")
        ->default_str("default_agent");
    
    app.add_option("-c,--capacity", config.memory.capacity, "Memory capacity")
        ->default_val(1000);
    
    app.add_option("-d,--decay", config.memory.decay_rate, "Memory decay rate")
        ->default_val(0.1);
    
    app.add_option("-t,--threshold", config.memory.retrieval_threshold, 
        "Memory retrieval threshold")
        ->default_val(0.5);
        
    app.add_option("-i,--input", input, "Input text")
        ->required();
}

int main(int argc, char** argv) {
    try {
        // Initialize logging
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [thread %t] %v");
        spdlog::info("Gloom Toolkit v{}", GLOOM_VERSION);

        // Parse command line arguments
        CLI::App app{"Gloom Toolkit - Intelligent Agent Framework"};
        AgentConfig config;
        std::string input;
        setup_cli(app, config, input);
        
        CLI11_PARSE(app, argc, argv);

        // Initialize agent
        spdlog::info("Initializing agent '{}'...", config.name);
        auto agent = std::make_unique<Agent>(config);

        // Initialize memory system
        spdlog::info("Initializing memory system...");
        auto memory = std::make_unique<Memory>(config.memory);

        // Initialize embedding generator
        spdlog::info("Initializing embedding system...");
        auto embedding_generator = std::make_unique<utils::EmbeddingGenerator>(384);

        // Process input
        spdlog::info("Processing input: '{}'", input);
        
        // Generate embedding for input
        auto embedding = embedding_generator->generate(input);
        
        // Store in memory
        MemoryEntry entry{
            .content = input,
            .embedding = embedding,
            .timestamp = std::time(nullptr),
            .importance = 0.8f
        };
        
        memory->store(entry);

        // Process with agent
        ProcessOptions options{
            .max_tokens = 100,
            .temperature = 0.7f
        };

        auto response = agent->process(input, options);

        // Output results
        std::cout << "\nAgent Response:\n" << response << std::endl;

        // Print memory statistics
        auto stats = memory->get_stats();
        spdlog::info("Memory Statistics:");
        spdlog::info("- Total entries: {}", stats.total_entries);
        spdlog::info("- Average importance: {:.2f}", stats.avg_importance);
        spdlog::info("- Memory usage: {:.2f}MB", stats.memory_usage_mb);

        return 0;
    }
    catch (const std::exception& e) {
        spdlog::error("Error: {}", e.what());
        return 1;
    }
}

#ifdef ENABLE_TESTS
int run_tests(int argc, char** argv) {
    // Initialize test framework
    doctest::Context context;
    context.setOption("abort-after", 5);  // Stop after 5 failed assertions
    context.applyCommandLine(argc, argv);
    
    int test_result = context.run();
    
    if (context.shouldExit()) {
        return test_result;
    }
    
    return test_result;
}
#endif

#ifdef ENABLE_BENCHMARKS
void run_benchmarks() {
    // Setup benchmark parameters
    const size_t num_iterations = 1000;
    const std::string test_input = "Test input for benchmarking";
    
    // Initialize components
    AgentConfig config;
    Agent agent(config);
    Memory memory(config.memory);
    utils::EmbeddingGenerator embedding_generator(384);
    
    // Benchmark embedding generation
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_iterations; ++i) {
            auto embedding = embedding_generator.generate(test_input);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count();
        
        spdlog::info("Embedding Generation Benchmark:");
        spdlog::info("- {} iterations in {}ms", num_iterations, duration);
        spdlog::info("- Average: {:.2f}ms per operation", 
            static_cast<float>(duration) / num_iterations);
    }
    
    // Benchmark memory operations
    {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < num_iterations; ++i) {
            MemoryEntry entry{
                .content = test_input,
                .embedding = embedding_generator.generate(test_input),
                .timestamp = std::time(nullptr),
                .importance = 0.8f
            };
            memory.store(entry);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count();
        
        spdlog::info("Memory Operation Benchmark:");
        spdlog::info("- {} iterations in {}ms", num_iterations, duration);
        spdlog::info("- Average: {:.2f}ms per operation", 
            static_cast<float>(duration) / num_iterations);
    }
}
#endif