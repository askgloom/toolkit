# Gloom Toolkit

<div align="center">
  <img src="https://raw.githubusercontent.com/askgloom/.github/refs/heads/main/images/banner.png" alt="Alone Labs Banner" width="100%" />
</div>

A modern C++ toolkit for building intelligent agents with sophisticated memory and planning capabilities.

## Overview

Gloom provides a comprehensive set of tools for developing AI agents with human-like memory systems, strategic planning, and adaptive behavior. Built with performance and flexibility in mind, it offers both high-level abstractions and low-level control.

## Features

### Memory Systems
- **Episodic Memory**: Temporal storage and retrieval of experience-based memories
- **Semantic Memory**: Graph-based knowledge representation with relationship tracking
- **Memory Store**: Generic memory management with importance-based retention

### Core Components
- Agent framework for building autonomous systems
- Environment abstractions for agent interaction
- State management and action planning
- Strategic goal planning and execution

## Installation

git clone https://github.com/yourusername/gloom.git
cd gloom
mkdir build && cd build
cmake ..
make

## Quick Start

### Creating an Agent with Memory

#include <gloom/core/agent.hpp>
#include <gloom/memory/episodic.hpp>
#include <gloom/memory/semantic.hpp>

int main() {
    using namespace gloom;
    
    // Initialize agent with memory systems
    Agent agent;
    memory::EpisodicMemory episodic(100);  // 100 episodes capacity
    memory::SemanticMemory semantic(1000);  // 1000 nodes capacity
    
    // Create an episodic memory
    auto episode_id = episodic.create_episode({
        {"location", "lab"},
        {"task", "experiment"}
    });
    
    // Add a memory to the episode
    episodic.add_memory(episode_id, {
        .content = "Observed unexpected behavior in test case 3",
        .importance = 0.8
    });
    
    return 0;
}

### Working with Semantic Memory

#include <gloom/memory/semantic.hpp>

void semantic_example() {
    using namespace gloom::memory;
    
    SemanticMemory semantic(1000);
    
    // Create nodes representing concepts
    auto cat_id = semantic.create_node("animal", {
        {"type", "mammal"},
        {"species", "cat"},
        {"domesticated", "true"}
    });
    
    auto dog_id = semantic.create_node("animal", {
        {"type", "mammal"},
        {"species", "dog"},
        {"domesticated", "true"}
    });
    
    // Create relationships between nodes
    semantic.add_relationship(cat_id, dog_id, 0.7);  // Strong relationship
    
    // Query related concepts
    auto related = semantic.get_related_nodes(cat_id, 0.5);
}

### Memory Search and Retrieval

#include <gloom/memory/episodic.hpp>

void search_example() {
    using namespace gloom::memory;
    
    EpisodicMemory memory(100);
    
    // Search for memories with specific context
    auto results = memory.search({
        .context = {{"location", "lab"}},
        .content = "unexpected",
        .start_time = std::chrono::system_clock::now() - std::chrono::hours(24)
    }, 10);  // Limit to 10 results
    
    for (const auto& [id, memories] : results) {
        for (const auto& memory : memories) {
            std::cout << "Memory: " << memory.content << "\n";
            std::cout << "Importance: " << memory.importance << "\n";
        }
    }
}

### Building a Custom Agent

#include <gloom/core/agent.hpp>

class CustomAgent : public gloom::Agent {
public:
    void perceive(const Environment& env) override {
        // Process environmental input
        auto state = env.get_state();
        
        // Store in episodic memory
        auto episode_id = episodic_memory.create_episode({
            {"state", state.to_string()},
            {"timestamp", std::to_string(time(nullptr))}
        });
    }
    
    Action decide() override {
        // Query memories to inform decision
        auto relevant_memories = episodic_memory.search({
            .content = "success",
            .limit = 5
        });
        
        // Use semantic memory for context
        auto context = semantic_memory.search({
            .concept = "current_goal"
        });
        
        // Make decision based on memories and context
        return plan_action(relevant_memories, context);
    }
    
private:
    memory::EpisodicMemory episodic_memory{1000};
    memory::SemanticMemory semantic_memory{1000};
};

## Requirements

- C++17 or later
- CMake 3.15+
- Modern compiler (GCC 7+, Clang 6+, MSVC 2019+)

## Documentation

Comprehensive documentation is available in the `docs/` directory:
- API Reference
- Architecture Guide
- Examples
- Best Practices

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Status

Project is in active development. API may change before v1.0 release.

## Contact

For questions and support:
- GitHub Issues: [Project Issues](https://github.com/askgloom/toolkit/issues)
- Email: team@askgloom.com
