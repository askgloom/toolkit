#pragma once

#include <gloom/core/agent.hpp>
#include <gloom/core/memory.hpp>
#include <gloom/utils/embeddings.hpp>
#include <random>
#include <vector>
#include <memory>
#include <functional>

namespace gloom {
namespace utils {

class AIGenerator {
public:
    // Configuration structure for evolution parameters
    struct GenerationConfig {
        size_t population_size = 100;
        size_t generations = 50;
        float mutation_rate = 0.1f;
        float crossover_rate = 0.7f;
        size_t tournament_size = 5;
        std::string fitness_metric = "accuracy";
        bool enable_elitism = true;
        size_t elite_count = 5;
        bool parallel_evaluation = true;
        size_t evaluation_threads = 4;
        
        // Validation
        bool validate() const {
            return population_size > 0 &&
                   generations > 0 &&
                   mutation_rate >= 0.0f && mutation_rate <= 1.0f &&
                   crossover_rate >= 0.0f && crossover_rate <= 1.0f &&
                   tournament_size > 0 && tournament_size <= population_size &&
                   elite_count <= population_size;
        }
    };

    // Genome structure representing an agent's genetic makeup
    struct AgentGenome {
        AgentConfig config;
        float fitness = 0.0f;
        std::vector<float> weights;
        std::unordered_map<std::string, float> metrics;
        
        bool operator<(const AgentGenome& other) const {
            return fitness < other.fitness;
        }
    };

    // Evolution statistics
    struct EvolutionStats {
        size_t generation;
        float best_fitness;
        float average_fitness;
        float worst_fitness;
        AgentGenome best_genome;
        std::vector<float> fitness_history;
        std::chrono::milliseconds evolution_time;
    };

    // Custom fitness function type
    using FitnessFunction = std::function<float(
        const Agent&,
        const std::vector<std::string>&
    )>;

    // Constructor
    explicit AIGenerator(const GenerationConfig& config);

    // Destructor
    virtual ~AIGenerator() = default;

    // Main evolution interface
    std::vector<AgentGenome> evolve(
        const std::vector<std::string>& training_data,
        const FitnessFunction& fitness_fn = nullptr
    );

    // Configuration methods
    void set_config(const GenerationConfig& config);
    const GenerationConfig& get_config() const;

    // Statistics and monitoring
    const EvolutionStats& get_stats() const;
    void reset_stats();

    // Genome manipulation
    AgentGenome create_genome() const;
    bool validate_genome(const AgentGenome& genome) const;
    void save_genome(const AgentGenome& genome, const std::string& path) const;
    AgentGenome load_genome(const std::string& path) const;

    // Event callbacks
    using GenerationCallback = std::function<void(const EvolutionStats&)>;
    void set_generation_callback(GenerationCallback callback);

protected:
    // Protected methods for inheritance
    virtual float evaluate_agent(
        const Agent& agent,
        const std::vector<std::string>& training_data
    );

    virtual std::pair<AgentGenome, AgentGenome> crossover(
        const AgentGenome& parent1,
        const AgentGenome& parent2
    );

    virtual void mutate(AgentGenome& genome);

private:
    // Private implementation details
    class Impl;
    std::unique_ptr<Impl> pimpl_;

    // Genetic algorithm core methods
    std::vector<AgentGenome> initialize_population();
    void evaluate_population(
        std::vector<AgentGenome>& population,
        const std::vector<std::string>& training_data
    );
    std::vector<AgentGenome> create_next_generation(
        const std::vector<AgentGenome>& current_pop
    );
    AgentGenome tournament_select(
        const std::vector<AgentGenome>& population
    );

    // Helper methods
    AgentConfig generate_random_config() const;
    std::vector<float> generate_random_weights() const;
    void update_stats(const std::vector<AgentGenome>& population);
    void report_progress() const;

    // Utility methods
    static float calculate_similarity(
        const std::vector<float>& vec1,
        const std::vector<float>& vec2
    );

    // Thread pool for parallel evaluation
    class ThreadPool;
    std::unique_ptr<ThreadPool> thread_pool_;

    // Member variables
    GenerationConfig config_;
    EvolutionStats stats_;
    std::mt19937 rng_;
    std::unique_ptr<Memory> memory_;
    EmbeddingGenerator embedding_generator_;
    GenerationCallback generation_callback_;

    // Constants
    static constexpr size_t DEFAULT_WEIGHT_SIZE = 100;
    static constexpr float MIN_TEMPERATURE = 0.1f;
    static constexpr float MAX_TEMPERATURE = 2.0f;
};

// Exception classes
class AIGeneratorError : public std::runtime_error {
public:
    explicit AIGeneratorError(const std::string& message)
        : std::runtime_error(message) {}
};

class InvalidConfigError : public AIGeneratorError {
public:
    explicit InvalidConfigError(const std::string& message)
        : AIGeneratorError(message) {}
};

class InvalidGenomeError : public AIGeneratorError {
public:
    explicit InvalidGenomeError(const std::string& message)
        : AIGeneratorError(message) {}
};

// Utility functions
namespace aigen_utils {
    // Serialization
    std::string serialize_genome(const AIGenerator::AgentGenome& genome);
    AIGenerator::AgentGenome deserialize_genome(const std::string& data);

    // Validation
    bool validate_config(const AIGenerator::GenerationConfig& config);
    bool validate_genome(const AIGenerator::AgentGenome& genome);

    // Metrics
    float calculate_diversity(const std::vector<AIGenerator::AgentGenome>& population);
    std::vector<float> analyze_fitness_landscape(
        const AIGenerator::AgentGenome& genome,
        float radius,
        size_t samples
    );
}

} // namespace utils
} // namespace gloom