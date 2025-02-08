#include "gloom/planning/planner.hpp"
#include "gloom/utils/logger.hpp"
#include <algorithm>
#include <queue>
#include <unordered_set>

namespace gloom {
namespace planning {

class Planner::Impl {
public:
    struct PlanNode {
        State current_state;
        std::vector<Action> actions;
        double cost;
        double heuristic;

        bool operator>(const PlanNode& other) const {
            return (cost + heuristic) > (other.cost + other.heuristic);
        }
    };

    explicit Impl(const PlannerConfig& config)
        : config_(config)
        , logger_("Planner") {}

    Plan create_plan(
        const State& initial_state,
        const Goal& goal,
        const std::vector<Action>& available_actions
    ) {
        logger_.info("Creating plan from initial state to goal");
        
        std::priority_queue<
            PlanNode,
            std::vector<PlanNode>,
            std::greater<PlanNode>
        > frontier;
        
        std::unordered_set<State, StateHash> explored;
        
        // Initialize search with start state
        frontier.push({
            initial_state,
            {},
            0.0,
            calculate_heuristic(initial_state, goal)
        });

        while (!frontier.empty() && explored.size() < config_.max_explored_states) {
            auto current = frontier.top();
            frontier.pop();

            if (goal.is_satisfied_by(current.current_state)) {
                logger_.info("Goal state reached, returning plan");
                return create_plan_from_actions(current.actions);
            }

            if (explored.find(current.current_state) != explored.end()) {
                continue;
            }

            explored.insert(current.current_state);

            // Explore possible actions
            for (const auto& action : available_actions) {
                if (!action.is_applicable(current.current_state)) {
                    continue;
                }

                State next_state = action.apply(current.current_state);
                double action_cost = action.get_cost();

                std::vector<Action> new_actions = current.actions;
                new_actions.push_back(action);

                frontier.push({
                    next_state,
                    new_actions,
                    current.cost + action_cost,
                    calculate_heuristic(next_state, goal)
                });
            }
        }

        logger_.warn("No plan found within constraints");
        return Plan(); // Empty plan indicates failure
    }

    double calculate_heuristic(const State& state, const Goal& goal) {
        // Basic heuristic based on goal distance
        return goal.distance_to(state);
    }

    Plan create_plan_from_actions(const std::vector<Action>& actions) {
        Plan plan;
        plan.actions = actions;
        plan.total_cost = std::accumulate(
            actions.begin(),
            actions.end(),
            0.0,
            [](double sum, const Action& action) {
                return sum + action.get_cost();
            }
        );
        return plan;
    }

    bool validate_plan(
        const Plan& plan,
        const State& initial_state,
        const Goal& goal
    ) {
        State current_state = initial_state;

        for (const auto& action : plan.actions) {
            if (!action.is_applicable(current_state)) {
                logger_.error("Plan validation failed: action not applicable");
                return false;
            }
            current_state = action.apply(current_state);
        }

        if (!goal.is_satisfied_by(current_state)) {
            logger_.error("Plan validation failed: goal not reached");
            return false;
        }

        return true;
    }

private:
    PlannerConfig config_;
    Logger logger_;
};

Planner::Planner(const PlannerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Planner::~Planner() = default;

Plan Planner::create_plan(
    const State& initial_state,
    const Goal& goal,
    const std::vector<Action>& available_actions
) {
    return pimpl_->create_plan(initial_state, goal, available_actions);
}

bool Planner::validate_plan(
    const Plan& plan,
    const State& initial_state,
    const Goal& goal
) {
    return pimpl_->validate_plan(plan, initial_state, goal);
}

void Planner::optimize_plan(Plan& plan) {
    // Remove redundant actions
    auto it = std::unique(
        plan.actions.begin(),
        plan.actions.end(),
        [](const Action& a1, const Action& a2) {
            return a1.cancels(a2);
        }
    );
    
    plan.actions.erase(it, plan.actions.end());

    // Recalculate total cost
    plan.total_cost = std::accumulate(
        plan.actions.begin(),
        plan.actions.end(),
        0.0,
        [](double sum, const Action& action) {
            return sum + action.get_cost();
        }
    );
}

} // namespace planning
} // namespace gloom