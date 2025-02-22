/**
 * @file bert.hpp
 * @brief BERT model implementation for Gloom toolkit
 * 
 * Provides BERT (Bidirectional Encoder Representations from Transformers)
 * model implementation with support for inference and fine-tuning.
 */

#ifndef GLOOM_MODELS_BERT_HPP
#define GLOOM_MODELS_BERT_HPP

#include <gloom/core/model.hpp>
#include <gloom/core/types.hpp>
#include <gloom/utils/tensor.hpp>

#include <memory>
#include <string>
#include <vector>

namespace gloom {
namespace models {

/**
 * @brief Configuration for BERT model
 */
struct BertConfig {
    size_t vocab_size = 30522;        ///< Vocabulary size
    size_t hidden_size = 768;         ///< Hidden layer size
    size_t num_hidden_layers = 12;    ///< Number of transformer layers
    size_t num_attention_heads = 12;  ///< Number of attention heads
    size_t intermediate_size = 3072;  ///< Intermediate layer size
    float hidden_dropout_prob = 0.1;  ///< Hidden layer dropout probability
    float attention_dropout_prob = 0.1; ///< Attention dropout probability
    size_t max_position_embeddings = 512; ///< Maximum sequence length
    size_t type_vocab_size = 2;       ///< Token type vocabulary size
    float layer_norm_eps = 1e-12;     ///< Layer normalization epsilon
};

/**
 * @brief BERT model implementation
 */
class BERT : public core::Model {
public:
    /**
     * @brief Construct BERT model
     * @param config Model configuration
     */
    explicit BERT(const BertConfig& config);

    /**
     * @brief Load model from file
     * @param path Path to model file
     * @return Success status
     */
    bool load(const std::string& path) override;

    /**
     * @brief Save model to file
     * @param path Path to save model
     * @return Success status
     */
    bool save(const std::string& path) const override;

    /**
     * @brief Run inference on input
     * @param input Input tensor
     * @return Output tensor
     */
    utils::Tensor forward(const utils::Tensor& input) override;

    /**
     * @brief Get model configuration
     * @return Current configuration
     */
    const BertConfig& config() const { return config_; }

    /**
     * @brief Create attention mask
     * @param input_ids Input token IDs
     * @return Attention mask tensor
     */
    utils::Tensor create_attention_mask(const std::vector<int64_t>& input_ids) const;

    /**
     * @brief Create position IDs
     * @param input_ids Input token IDs
     * @return Position IDs tensor
     */
    utils::Tensor create_position_ids(const std::vector<int64_t>& input_ids) const;

    /**
     * @brief Encode text sequence
     * @param text Input text
     * @return Encoded representation
     */
    utils::Tensor encode(const std::string& text);

    /**
     * @brief Get embedding layer output
     * @param input_ids Input token IDs
     * @param token_type_ids Token type IDs
     * @param position_ids Position IDs
     * @return Embedding tensor
     */
    utils::Tensor get_embeddings(
        const std::vector<int64_t>& input_ids,
        const std::vector<int64_t>& token_type_ids,
        const std::vector<int64_t>& position_ids
    ) const;

protected:
    /**
     * @brief Initialize model parameters
     */
    void initialize_parameters();

    /**
     * @brief Apply self attention
     * @param hidden_states Input hidden states
     * @param attention_mask Attention mask
     * @return Updated hidden states
     */
    utils::Tensor self_attention(
        const utils::Tensor& hidden_states,
        const utils::Tensor& attention_mask
    ) const;

    /**
     * @brief Apply feed forward network
     * @param hidden_states Input hidden states
     * @return Updated hidden states
     */
    utils::Tensor feed_forward(const utils::Tensor& hidden_states) const;

private:
    BertConfig config_;                      ///< Model configuration
    std::vector<utils::Tensor> weights_;     ///< Model weights
    std::vector<utils::Tensor> biases_;      ///< Model biases
    
    utils::Tensor word_embeddings_;          ///< Word embedding matrix
    utils::Tensor position_embeddings_;      ///< Position embedding matrix
    utils::Tensor token_type_embeddings_;    ///< Token type embedding matrix
    
    utils::Tensor layer_norm_weight_;        ///< Layer normalization weight
    utils::Tensor layer_norm_bias_;          ///< Layer normalization bias

    bool is_training_ = false;               ///< Training mode flag
};

} // namespace models
} // namespace gloom

#endif // GLOOM_MODELS_BERT_HPP