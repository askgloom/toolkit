#include "gloom/planning/goals.hpp"
#include "gloom/utils/logger.hpp"
#include <algorithm>
#include <cmath>

namespace gloom {
namespace planning {

class Goal::Impl {
public:
    explicit Impl(const GoalConfig& config)
        : config_(config)
        , logger_("Goal")
        , conditions_()
        , weights_()
        , threshold_(0.95) {}

    void add_condition(
        const GoalCondition& condition,
        double weight = 1.0
    ) {
        conditions_.push_back(condition);
        weights_.push_back(weight);
        normalize_weights();
        logger_.debug("Added goal condition with weight: " + std::to_string(weight));
    }

    bool is_satisfied_by(const State& state) const {
        if (conditions_.empty()) {
            logger_.warn("No conditions set for goal");
            return false;
        }

        double total_satisfaction = calculate_satisfaction(state);
        bool satisfied = total_satisfaction >= threshold_;

        logger_.debug(
            "Goal satisfaction level: " + 
            std::to_string(total_satisfaction) +
            (satisfied ? " (satisfied)" : " (not satisfied)")
        );

        return satisfied;
    }

    double distance_to(const State& state) const {
        if (conditions_.empty()) {
            logger_.warn("No conditions set for goal distance calculation");
            return std::numeric_limits<double>::infinity();
        }

        double total_distance = 0.0;
        double total_weight = 0.0;

        for (size_t i = 0; i < conditions_.size(); ++i) {
            const auto& condition = conditions_[i];
            double weight = weights_[i];
            
            double condition_distance = condition.distance_to(state);
            total_distance += weight * condition_distance;
            total_weight += weight;
        }

        return total_weight > 0.0 ? total_distance / total_weight : 
                                  std::numeric_limits<double>::infinity();
    }

    std::vector<GoalCondition> get_unsatisfied_conditions(
        const State& state
    ) const {
        std::vector<GoalCondition> unsatisfied;
        
        for (const auto& condition : conditions_) {
            if (!condition.is_satisfied_by(state)) {
                unsatisfied.push_back(condition);
            }
        }
        
        return unsatisfied;
    }

    void set_threshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            logger_.error(
                "Invalid threshold value: " + std::to_string(threshold) +
                ". Must be between 0.0 and 1.0"
            );
            return;
        }
        
        threshold_ = threshold;
        logger_.info("Set satisfaction threshold to: " + std::to_string(threshold));
    }

    double get_threshold() const {
        return threshold_;
    }

    size_t condition_count() const {
        return conditions_.size();
    }

    void clear() {
        conditions_.clear();
        weights_.clear();
        logger_.info("Cleared all goal conditions");
    }

private:
    GoalConfig config_;
    Logger logger_;
    std::vector<GoalCondition> conditions_;
    std::vector<double> weights_;
    double threshold_;

    void normalize_weights() {
        double sum = std::accumulate(
            weights_.begin(),
            weights_.end(),
            0.0
        );
        
        if (sum > 0.0) {
            std::transform(
                weights_.begin(),
                weights_.end(),
                weights_.begin(),
                [sum](double w) { return w / sum; }
            );
        }
    }

    double calculate_satisfaction(const State& state) const {
        double total_satisfaction = 0.0;
        double total_weight = 0.0;

        for (size_t i = 0; i < conditions_.size(); ++i) {
            const auto& condition = conditions_[i];
            double weight = weights_[i];
            
            if (condition.is_satisfied_by(state)) {
                total_satisfaction += weight;
            }
            total_weight += weight;
        }

        return total_weight > 0.0 ? total_satisfaction / total_weight : 0.0;
    }
};

Goal::Goal(const GoalConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Goal::~Goal() = default;

void Goal::add_condition(const GoalCondition& condition, double weight) {
    pimpl_->add_condition(condition, weight);
}

bool Goal::is_satisfied_by(const State& state) const {
    return pimpl_->is_satisfied_by(state);
}

double Goal::distance_to(const State& state) const {
    return pimpl_->distance_to(state);
}

std::vector<GoalCondition> Goal::get_unsatisfied_conditions(
    const State& state
) const {
    return pimpl_->get_unsatisfied_conditions(state);
}

void Goal::set_threshold(double threshold) {
    pimpl_->set_threshold(threshold);
}

double Goal::get_threshold() const {
    return pimpl_->get_threshold();
}

size_t Goal::condition_count() const {
    return pimpl_->condition_count();
}

void Goal::clear() {
    pimpl_->clear();
}

// GoalCondition implementation
bool GoalCondition::is_satisfied_by(const State& state) const {
    return evaluate(state) >= threshold_;
}

double GoalCondition::distance_to(const State& state) const {
    double satisfaction = evaluate(state);
    return std::max(0.0, threshold_ - satisfaction);
}

} // namespace planning
} // namespace gloom