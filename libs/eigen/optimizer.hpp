#pragma once

#include <gloom/core/agent.hpp>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <chrono>

namespace gloom {
namespace utils {

// Forward declarations
class Agent;

enum class OptimizationAlgorithm {
    BAYESIAN,
    PARTICLE_SWARM,
    GRID_SEARCH,
    SIMULATED_ANNEALING,
    EVOLUTIONARY,
    GRADIENT_DESCENT
};

struct Parameter {
    std::string name;
    float min_value;
    float max_value;
    float default_value;
    std::string description;
    bool is_discrete = false;
    std::vector<float> discrete_values;
};

struct OptimizerConfig {
    OptimizationAlgorithm algorithm = OptimizationAlgorithm::BAYESIAN;
    size_t max_iterations = 100;
    size_t population_size = 50;
    float convergence_threshold = 1e-6f;
    bool enable_parallel = true;
    size_t num_threads = 4;
    
    // Algorithm-specific parameters
    struct {
        float acquisition_function_kappa = 2.576f;
        size_t num_random_samples = 10;
        bool use_ucb = true;
    } bayesian;
    
    struct {
        float inertia_weight = 0.729f;
        float cognitive_param = 1.49445f;
        float social_param = 1.49445f;
        float velocity_clamp = 0.1f;
    } particle_swarm;
    
    struct {
        float initial_temperature = 1.0f;
        float cooling_rate = 0.95f;
        size_t steps_per_temp = 50;
    } simulated_annealing;
    
    struct {
        float mutation_rate = 0.1f;
        float crossover_rate = 0.7f;
        size_t tournament_size = 5;
        bool elitism = true;
    } evolutionary;
    
    struct {
        float learning_rate = 0.01f;
        float momentum = 0.9f;
        float gradient_clip = 5.0f;
    } gradient_descent;
};

struct OptimizationResult {
    std::vector<float> best_parameters;
    float best_score;
    size_t iterations_used;
    std::chrono::milliseconds optimization_time;
    std::vector<float> convergence_history;
    std::unordered_map<std::string, float> metrics;
};

// Custom evaluation function type
using EvaluationFunction = std::function<float(
    const Agent&,
    const std::vector<std::string>&
)>;

class OptimizerError : public std::runtime_error {
public:
    explicit OptimizerError(const std::string& message)
        : std::runtime_error(message) {}
};

class Optimizer {
public:
    explicit Optimizer(const OptimizerConfig& config);
    ~Optimizer();

    // Prevent copying
    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = delete;

    // Core optimization methods
    void optimize(Agent& agent, const std::vector<std::string>& training_data);
    
    OptimizationResult optimize_with_custom_evaluation(
        Agent& agent,
        const std::vector<std::string>& training_data,
        const EvaluationFunction& eval_fn
    );

    // Configuration methods
    void set_config(const OptimizerConfig& config);
    const OptimizerConfig& get_config() const;
    void set_parameters(const std::vector<Parameter>& parameters);
    
    // Progress monitoring
    using ProgressCallback = std::function<void(
        size_t iteration,
        float best_score,
        const std::vector<float>& current_params
    )>;
    void set_progress_callback(ProgressCallback callback);

    // Results and analysis
    OptimizationResult get_last_result() const;
    std::vector<float> get_parameter_importance() const;
    std::vector<std::pair<float, float>> get_parameter_sensitivity(
        size_t param_index
    ) const;

    // Utility methods
    static bool validate_config(const OptimizerConfig& config);
    static bool validate_parameters(const std::vector<Parameter>& parameters);
    
    // Parameter space exploration
    std::vector<std::vector<float>> generate_random_samples(
        size_t num_samples
    ) const;
    
    std::vector<std::vector<float>> generate_grid_samples(
        size_t points_per_dim
    ) const;

protected:
    // Protected methods for inheritance
    virtual float evaluate_parameters(
        Agent& agent,
        const std::vector<std::string>& training_data,
        const std::vector<float>& parameters
    );

    virtual void apply_parameters(
        Agent& agent,
        const std::vector<float>& parameters
    );

    virtual float evaluate_response(
        const std::string& response,
        const std::string& expected
    );

private:
    // Private implementation
    class Impl;
    std::unique_ptr<Impl> pimpl_;

    // Optimization algorithms
    class BayesianOptimizer;
    class ParticleSwarmOptimizer;
    class GridSearchOptimizer;
    class SimulatedAnnealingOptimizer;
    class EvolutionaryOptimizer;
    class GradientDescentOptimizer;

    // Gaussian Process for Bayesian optimization
    class GaussianProcess {
    public:
        explicit GaussianProcess(size_t input_dim);
        std::vector<float> sample_next_point(
            const std::vector<std::vector<float>>& X,
            const std::vector<float>& y
        );
    private:
        size_t input_dim_;
        // Additional GP implementation details
    };

    // Utility classes
    class ParameterSpace;
    class OptimizationMetrics;
    class ThreadPool;
};

// Utility functions
namespace optimizer_utils {
    // Parameter space utilities
    std::vector<float> normalize_parameters(
        const std::vector<float>& params,
        const std::vector<Parameter>& parameter_specs
    );
    
    std::vector<float> denormalize_parameters(
        const std::vector<float>& normalized_params,
        const std::vector<Parameter>& parameter_specs
    );

    // Evaluation utilities
    float calculate_similarity(
        const std::vector<float>& vec1,
        const std::vector<float>& vec2
    );
    
    float calculate_diversity(
        const std::vector<std::vector<float>>& population
    );

    // Analysis utilities
    std::vector<float> compute_parameter_importance(
        const std::vector<std::vector<float>>& parameters,
        const std::vector<float>& scores
    );
    
    std::vector<std::pair<float, float>> analyze_parameter_sensitivity(
        const std::function<float(const std::vector<float>&)>& objective,
        const std::vector<float>& base_point,
        size_t param_index,
        float range = 0.1f,
        size_t num_points = 20
    );
}

} // namespace utils
} // namespace gloom