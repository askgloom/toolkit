/**
 * @file transformer.hpp
 * @brief Base transformer implementation for Gloom toolkit
 * 
 * Provides core transformer architecture components that can be
 * extended by specific model implementations like BERT and LLAMA.
 */

#ifndef GLOOM_MODELS_TRANSFORMER_HPP
#define GLOOM_MODELS_TRANSFORMER_HPP

#include <gloom/core/model.hpp>
#include <gloom/core/types.hpp>
#include <gloom/utils/tensor.hpp>

#include <memory>
#include <string>
#include <vector>
#include <optional>

namespace gloom {
namespace models {

/**
 * @brief Configuration for transformer models
 */
struct TransformerConfig {
    size_t vocab_size = 50257;        ///< Vocabulary size
    size_t hidden_size = 768;         ///< Hidden layer size
    size_t num_layers = 12;           ///< Number of transformer layers
    size_t num_attention_heads = 12;  ///< Number of attention heads
    size_t max_position_embeddings = 2048; ///< Maximum sequence length
    float dropout_prob = 0.1;         ///< Dropout probability
    float attention_dropout_prob = 0.1; ///< Attention dropout probability
    float layer_norm_eps = 1e-12;     ///< Layer normalization epsilon
    bool use_bias = true;             ///< Whether to use bias in linear layers
    std::string activation_fn = "gelu"; ///< Activation function
    bool use_rotary_embeddings = false; ///< Whether to use rotary embeddings
    bool use_flash_attention = true;   ///< Whether to use flash attention
    size_t intermediate_size = 3072;   ///< Size of feed-forward intermediate layer
};

/**
 * @brief Base transformer model implementation
 */
class Transformer : public core::Model {
public:
    /**
     * @brief Construct transformer model
     * @param config Model configuration
     */
    explicit Transformer(const TransformerConfig& config);
    virtual ~Transformer() = default;

    /**
     * @brief Get model configuration
     * @return Current configuration
     */
    const TransformerConfig& config() const { return config_; }

protected:
    /**
     * @brief Multi-head attention implementation
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param mask Attention mask
     * @return Attention output
     */
    utils::Tensor multi_head_attention(
        const utils::Tensor& query,
        const utils::Tensor& key,
        const utils::Tensor& value,
        const std::optional<utils::Tensor>& mask = std::nullopt
    ) const;

    /**
     * @brief Feed forward network implementation
     * @param input Input tensor
     * @return Processed tensor
     */
    utils::Tensor feed_forward_network(const utils::Tensor& input) const;

    /**
     * @brief Layer normalization
     * @param input Input tensor
     * @param eps Epsilon value
     * @return Normalized tensor
     */
    utils::Tensor layer_norm(
        const utils::Tensor& input,
        float eps = 1e-12
    ) const;

    /**
     * @brief Apply activation function
     * @param input Input tensor
     * @return Activated tensor
     */
    utils::Tensor activate(const utils::Tensor& input) const;

    /**
     * @brief Apply rotary position embeddings
     * @param input Input tensor
     * @param position_ids Position IDs
     * @return Embedded tensor
     */
    utils::Tensor apply_rotary_embeddings(
        const utils::Tensor& input,
        const utils::Tensor& position_ids
    ) const;

    /**
     * @brief Flash attention implementation
     * @param query Query tensor
     * @param key Key tensor
     * @param value Value tensor
     * @param mask Attention mask
     * @return Attention output
     */
    utils::Tensor flash_attention(
        const utils::Tensor& query,
        const utils::Tensor& key,
        const utils::Tensor& value,
        const std::optional<utils::Tensor>& mask = std::nullopt
    ) const;

    /**
     * @brief Create causal mask
     * @param size Sequence length
     * @return Causal mask tensor
     */
    utils::Tensor create_causal_mask(size_t size) const;

    /**
     * @brief Apply dropout
     * @param input Input tensor
     * @param prob Dropout probability
     * @return Output tensor
     */
    utils::Tensor dropout(
        const utils::Tensor& input,
        float prob
    ) const;

    /**
     * @brief Initialize model parameters
     */
    virtual void initialize_parameters();

    /**
     * @brief Get attention scores
     * @param query Query tensor
     * @param key Key tensor
     * @param mask Attention mask
     * @return Attention scores
     */
    utils::Tensor get_attention_scores(
        const utils::Tensor& query,
        const utils::Tensor& key,
        const std::optional<utils::Tensor>& mask = std::nullopt
    ) const;

protected:
    TransformerConfig config_;                    ///< Model configuration
    std::vector<utils::Tensor> layer_weights_;    ///< Layer weights
    std::vector<utils::Tensor> layer_biases_;     ///< Layer biases
    utils::Tensor position_embeddings_;           ///< Position embeddings
    utils::Tensor token_embeddings_;              ///< Token embeddings
    bool is_training_ = false;                    ///< Training mode flag

private:
    /**
     * @brief Split heads for attention
     * @param tensor Input tensor
     * @param num_heads Number of attention heads
     * @return Split tensor
     */
    utils::Tensor split_heads(
        const utils::Tensor& tensor,
        size_t num_heads
    ) const;

    /**
     * @brief Merge heads after attention
     * @param tensor Input tensor
     * @return Merged tensor
     */
    utils::Tensor merge_heads(const utils::Tensor& tensor) const;
};

} // namespace models
} // namespace gloom

#endif // GLOOM_MODELS_TRANSFORMER_HPP