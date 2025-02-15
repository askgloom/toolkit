#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <gloom/utils/embeddings.hpp>
#include <vector>
#include <string>

using namespace gloom::utils;
using Catch::Matchers::Approx;

TEST_CASE("Embeddings basic functionality", "[embeddings]") {
    EmbeddingGenerator generator(384); // Standard dimension

    SECTION("Generate embeddings from text") {
        std::string input = "Test sentence for embedding generation";
        auto embedding = generator.generate(input);

        REQUIRE(embedding.size() == 384);
        REQUIRE_FALSE(embedding.empty());
        
        // Check if values are normalized
        float sum_squared = 0.0f;
        for (const auto& val : embedding) {
            sum_squared += val * val;
        }
        REQUIRE(std::sqrt(sum_squared) == Approx(1.0).margin(1e-6));
    }

    SECTION("Batch embedding generation") {
        std::vector<std::string> inputs = {
            "First test sentence",
            "Second test sentence",
            "Third test sentence"
        };

        auto embeddings = generator.generate_batch(inputs);

        REQUIRE(embeddings.size() == 3);
        for (const auto& embedding : embeddings) {
            REQUIRE(embedding.size() == 384);
        }
    }

    SECTION("Empty input handling") {
        std::string empty_input = "";
        REQUIRE_THROWS_AS(generator.generate(empty_input), EmbeddingError);
    }
}

TEST_CASE("Embedding similarity calculations", "[embeddings]") {
    EmbeddingGenerator generator(384);

    SECTION("Similar text comparison") {
        std::string text1 = "The quick brown fox";
        std::string text2 = "The fast brown fox";

        auto embedding1 = generator.generate(text1);
        auto embedding2 = generator.generate(text2);

        float similarity = calculate_similarity(embedding1, embedding2);
        REQUIRE(similarity > 0.8f); // High similarity expected
        REQUIRE(similarity <= 1.0f);
    }

    SECTION("Dissimilar text comparison") {
        std::string text1 = "The quick brown fox";
        std::string text2 = "Completely different text";

        auto embedding1 = generator.generate(text1);
        auto embedding2 = generator.generate(text2);

        float similarity = calculate_similarity(embedding1, embedding2);
        REQUIRE(similarity < 0.5f); // Low similarity expected
        REQUIRE(similarity >= 0.0f);
    }
}

TEST_CASE("Embedding persistence", "[embeddings]") {
    EmbeddingGenerator generator(384);
    std::string test_text = "Test sentence for persistence";

    SECTION("Save and load embeddings") {
        auto original_embedding = generator.generate(test_text);
        
        // Save embedding
        std::string filename = "test_embedding.bin";
        REQUIRE_NOTHROW(save_embedding(original_embedding, filename));

        // Load embedding
        auto loaded_embedding = load_embedding(filename);
        
        REQUIRE(loaded_embedding.size() == original_embedding.size());
        for (size_t i = 0; i < original_embedding.size(); ++i) {
            REQUIRE(loaded_embedding[i] == Approx(original_embedding[i]));
        }

        // Cleanup
        std::remove(filename.c_str());
    }

    SECTION("Handle invalid file loading") {
        REQUIRE_THROWS_AS(load_embedding("nonexistent_file.bin"), EmbeddingError);
    }
}

TEST_CASE("Embedding model configuration", "[embeddings]") {
    SECTION("Custom dimension configuration") {
        EmbeddingGenerator generator(512); // Custom dimension
        std::string input = "Test sentence";
        auto embedding = generator.generate(input);
        REQUIRE(embedding.size() == 512);
    }

    SECTION("Invalid dimension handling") {
        REQUIRE_THROWS_AS(EmbeddingGenerator(0), std::invalid_argument);
        REQUIRE_THROWS_AS(EmbeddingGenerator(-1), std::invalid_argument);
    }

    SECTION("Model parameters configuration") {
        EmbeddingConfig config{
            .dimension = 384,
            .normalize = true,
            .pooling_strategy = PoolingStrategy::MEAN
        };

        EmbeddingGenerator generator(config);
        std::string input = "Test sentence";
        auto embedding = generator.generate(input);
        REQUIRE(embedding.size() == 384);
    }
}

TEST_CASE("Embedding performance benchmarks", "[embeddings][!benchmark]") {
    EmbeddingGenerator generator(384);
    std::string long_text = std::string(1000, 'a'); // 1000 character string

    SECTION("Generation speed") {
        auto start = std::chrono::high_resolution_clock::now();
        auto embedding = generator.generate(long_text);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        REQUIRE(duration.count() < 1000); // Should complete within 1 second
    }

    SECTION("Batch processing efficiency") {
        std::vector<std::string> batch(100, "Test sentence"); // 100 identical sentences

        auto start = std::chrono::high_resolution_clock::now();
        auto embeddings = generator.generate_batch(batch);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        REQUIRE(duration.count() < 5000); // Should complete within 5 seconds
    }
}

TEST_CASE("Embedding error handling", "[embeddings]") {
    EmbeddingGenerator generator(384);

    SECTION("Invalid input handling") {
        REQUIRE_THROWS_AS(generator.generate(std::string()), EmbeddingError);
        REQUIRE_THROWS_AS(generator.generate(std::string(10000, 'a')), EmbeddingError);
    }

    SECTION("Invalid similarity calculation") {
        std::vector<float> embedding1 = generator.generate("Test");
        std::vector<float> embedding2(256, 0.0f); // Different dimension

        REQUIRE_THROWS_AS(
            calculate_similarity(embedding1, embedding2),
            std::invalid_argument
        );
    }
}