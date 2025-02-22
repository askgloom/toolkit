#include <gloom/models/llama.hpp>
#include <gloom/core/context.hpp>
#include <gloom/utils/tensor.hpp>
#include <gloom/utils/logger.hpp>

namespace gloom {
namespace models {

LLAMA::LLAMA(const LlamaConfig& config) : config_(config) {
    initialize_parameters();
}

bool LLAMA::load(const std::string& path) {
    try {
        // Load model weights and parameters
        utils::Logger::info("Loading LLAMA model from: {}", path);
        
        // Initialize tokenizer
        tokenizer_ = std::make_unique<utils::Tokenizer>(
            path + "/tokenizer.model"
        );

        // Load model weights
        weights_.clear();
        for (size_t i = 0; i < config_.num_layers; ++i) {
            auto layer_path = path + "/layer_" + std::to_string(i);
            weights_.push_back(utils::Tensor::load(layer_path));
        }

        // Load embeddings
        word_embeddings_ = utils::Tensor::load(path + "/embeddings.bin");
        
        utils::Logger::info("LLAMA model loaded successfully");
        return true;
    }
    catch (const std::exception& e) {
        utils::Logger::error("Failed to load LLAMA model: {}", e.what());
        return false;
    }
}

bool LLAMA::save(const std::string& path) const {
    try {
        // Save model weights and parameters
        utils::Logger::info("Saving LLAMA model to: {}", path);
        
        // Save tokenizer
        tokenizer_->save(path + "/tokenizer.model");

        // Save weights
        for (size_t i = 0; i < weights_.size(); ++i) {
            auto layer_path = path + "/layer_" + std::to_string(i);
            weights_[i].save(layer_path);
        }

        // Save embeddings
        word_embeddings_.save(path + "/embeddings.bin");
        
        utils::Logger::info("LLAMA model saved successfully");
        return true;
    }
    catch (const std::exception& e) {
        utils::Logger::error("Failed to save LLAMA model: {}", e.what());
        return false;
    }
}

utils::Tensor LLAMA::forward(const utils::Tensor& input) {
    // Input shape validation
    if (input.dims() != 2) {
        throw std::runtime_error("Input tensor must be 2-dimensional");
    }

    // Get embeddings
    auto hidden_states = get_embeddings(input);

    // Process through transformer layers
    for (size_t i = 0; i < config_.num_layers; ++i) {
        // Self attention
        auto attention_output = self_attention(
            hidden_states,
            create_attention_mask(input)
        );

        // Add & normalize
        hidden_states = layer_norm(
            hidden_states + attention_output
        );

        // Feed forward
        auto ff_output = feed_forward(hidden_states);

        // Add & normalize
        hidden_states = layer_norm(
            hidden_states + ff_output
        );
    }

    return hidden_states;
}

utils::Tensor LLAMA::generate(
    const std::string& prompt,
    size_t max_length,
    float temperature
) {
    // Tokenize input
    auto tokens = tokenizer_->encode(prompt);
    
    // Initialize context
    core::Context ctx;
    ctx.set_temperature(temperature);
    
    utils::Tensor input_tensor = utils::Tensor::from_vector(tokens);
    
    // Generate tokens
    for (size_t i = 0; i < max_length; ++i) {
        // Forward pass
        auto logits = forward(input_tensor);
        
        // Sample next token
        auto next_token = sample_token(logits, temperature);
        
        // Append to input
        tokens.push_back(next_token);
        input_tensor = utils::Tensor::from_vector(tokens);
        
        // Check for end of sequence
        if (next_token == tokenizer_->eos_token_id()) {
            break;
        }
    }
    
    return input_tensor;
}

void LLAMA::initialize_parameters() {
    utils::Logger::info("Initializing LLAMA parameters");
    
    // Initialize weights with scaled normal distribution
    float scale = std::sqrt(2.0f / (config_.hidden_size + config_.vocab_size));
    
    weights_.clear();
    for (size_t i = 0; i < config_.num_layers; ++i) {
        weights_.push_back(utils::Tensor::randn({
            config_.hidden_size,
            config_.hidden_size
        }, scale));
    }

    // Initialize embeddings
    word_embeddings_ = utils::Tensor::randn({
        config_.vocab_size,
        config_.hidden_size
    }, scale);
}

utils::Tensor LLAMA::layer_norm(const utils::Tensor& input) const {
    return (input - input.mean()) / (input.std() + config_.layer_norm_eps);
}

int64_t LLAMA::sample_token(const utils::Tensor& logits, float temperature) const {
    // Apply temperature
    auto scaled_logits = logits / temperature;
    
    // Convert to probabilities
    auto probs = scaled_logits.softmax();
    
    // Sample from distribution
    return probs.multinomial(1)[0];
}

} // namespace models
} // namespace gloom