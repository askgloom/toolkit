#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <gloom/utils/vectors.hpp>
#include <random>
#include <algorithm>

using namespace gloom::utils;
using Catch::Matchers::WithinAbs;
using Vector = std::vector<float>;

TEST_CASE("Vector operations", "[vectors]") {
    SECTION("Vector normalization") {
        Vector vec = {1.0f, 2.0f, 3.0f, 4.0f};
        auto normalized = normalize(vec);

        // Check length is 1
        float length = 0.0f;
        for (const auto& v : normalized) {
            length += v * v;
        }
        REQUIRE(std::sqrt(length) == Approx(1.0f).margin(1e-6));

        // Check direction preserved
        float ratio = vec[0] / normalized[0];
        for (size_t i = 1; i < vec.size(); ++i) {
            REQUIRE(vec[i] / normalized[i] == Approx(ratio).margin(1e-6));
        }
    }

    SECTION("Dot product") {
        Vector v1 = {1.0f, 2.0f, 3.0f};
        Vector v2 = {4.0f, 5.0f, 6.0f};
        
        float dot = dot_product(v1, v2);
        REQUIRE(dot == Approx(32.0f));  // 1*4 + 2*5 + 3*6 = 32
    }

    SECTION("Cosine similarity") {
        Vector v1 = {1.0f, 0.0f, 0.0f};
        Vector v2 = {1.0f, 1.0f, 0.0f};
        
        float similarity = cosine_similarity(v1, v2);
        REQUIRE(similarity == Approx(1.0f / std::sqrt(2.0f)));
    }
}

TEST_CASE("Vector index operations", "[vectors]") {
    VectorIndex index(384); // 384-dimensional vectors

    SECTION("Adding and retrieving vectors") {
        Vector vec1(384, 0.1f);
        Vector vec2(384, 0.2f);

        index.add("id1", vec1);
        index.add("id2", vec2);

        REQUIRE(index.size() == 2);
        
        auto retrieved = index.get("id1");
        REQUIRE(retrieved.has_value());
        REQUIRE(retrieved->size() == 384);
        REQUIRE(retrieved.value() == vec1);
    }

    SECTION("Nearest neighbor search") {
        // Create random vectors
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // Add 100 random vectors
        for (int i = 0; i < 100; ++i) {
            Vector vec(384);
            for (auto& v : vec) {
                v = dist(gen);
            }
            vec = normalize(vec);
            index.add("id" + std::to_string(i), vec);
        }

        // Search for nearest neighbors
        Vector query(384);
        for (auto& v : query) {
            v = dist(gen);
        }
        query = normalize(query);

        auto results = index.search(query, 10);
        REQUIRE(results.size() == 10);
        
        // Check results are sorted by similarity
        for (size_t i = 1; i < results.size(); ++i) {
            REQUIRE(results[i-1].similarity >= results[i].similarity);
        }
    }
}

TEST_CASE("Vector serialization", "[vectors]") {
    SECTION("Binary serialization") {
        Vector original = {1.0f, 2.0f, 3.0f, 4.0f};
        std::string filename = "test_vector.bin";

        REQUIRE_NOTHROW(save_vector(original, filename));
        auto loaded = load_vector(filename);

        REQUIRE(loaded == original);
        std::remove(filename.c_str());
    }

    SECTION("JSON serialization") {
        VectorMetadata metadata{
            .dimension = 384,
            .normalized = true,
            .created_at = std::time(nullptr)
        };
        Vector vec(384, 0.1f);

        std::string json = serialize_vector_json(vec, metadata);
        auto [loaded_vec, loaded_metadata] = deserialize_vector_json(json);

        REQUIRE(loaded_vec == vec);
        REQUIRE(loaded_metadata.dimension == metadata.dimension);
        REQUIRE(loaded_metadata.normalized == metadata.normalized);
    }
}

TEST_CASE("Vector batch operations", "[vectors]") {
    SECTION("Batch normalization") {
        std::vector<Vector> vectors = {
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f},
            {7.0f, 8.0f, 9.0f}
        };

        auto normalized = normalize_batch(vectors);
        REQUIRE(normalized.size() == vectors.size());

        for (const auto& vec : normalized) {
            float length = std::sqrt(std::inner_product(
                vec.begin(), vec.end(), vec.begin(), 0.0f
            ));
            REQUIRE(length == Approx(1.0f).margin(1e-6));
        }
    }

    SECTION("Batch similarity computation") {
        std::vector<Vector> vectors = {
            normalize({1.0f, 0.0f, 0.0f}),
            normalize({1.0f, 1.0f, 0.0f}),
            normalize({0.0f, 1.0f, 0.0f})
        };

        auto similarity_matrix = compute_similarity_matrix(vectors);
        REQUIRE(similarity_matrix.size() == vectors.size());
        REQUIRE(similarity_matrix[0].size() == vectors.size());

        // Check diagonal is 1
        for (size_t i = 0; i < vectors.size(); ++i) {
            REQUIRE(similarity_matrix[i][i] == Approx(1.0f));
        }

        // Check symmetry
        for (size_t i = 0; i < vectors.size(); ++i) {
            for (size_t j = 0; j < vectors.size(); ++j) {
                REQUIRE(similarity_matrix[i][j] == 
                       Approx(similarity_matrix[j][i]).margin(1e-6));
            }
        }
    }
}

TEST_CASE("Vector error handling", "[vectors]") {
    SECTION("Invalid vector operations") {
        Vector v1 = {1.0f, 2.0f};
        Vector v2 = {1.0f, 2.0f, 3.0f};

        REQUIRE_THROWS_AS(dot_product(v1, v2), std::invalid_argument);
        REQUIRE_THROWS_AS(cosine_similarity(v1, v2), std::invalid_argument);
    }

    SECTION("Zero vector handling") {
        Vector zero_vec = {0.0f, 0.0f, 0.0f};
        REQUIRE_THROWS_AS(normalize(zero_vec), std::domain_error);
    }

    SECTION("Index error handling") {
        VectorIndex index(3);
        Vector vec = {1.0f, 2.0f, 3.0f};
        
        index.add("id1", vec);
        REQUIRE_THROWS_AS(index.add("id1", vec), std::runtime_error); // Duplicate ID
        REQUIRE_THROWS_AS(index.get("nonexistent"), std::out_of_range);
    }
}

TEST_CASE("Vector performance benchmarks", "[vectors][!benchmark]") {
    const size_t dim = 384;
    const size_t num_vectors = 10000;
    
    std::vector<Vector> vectors;
    vectors.reserve(num_vectors);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Generate test vectors
    for (size_t i = 0; i < num_vectors; ++i) {
        Vector vec(dim);
        for (auto& v : vec) {
            v = dist(gen);
        }
        vectors.push_back(normalize(vec));
    }

    SECTION("Batch normalization performance") {
        auto start = std::chrono::high_resolution_clock::now();
        auto normalized = normalize_batch(vectors);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start
        ).count();

        REQUIRE(duration < 1000); // Should complete within 1 second
    }

    SECTION("Index search performance") {
        VectorIndex index(dim);
        
        // Build index
        for (size_t i = 0; i < vectors.size(); ++i) {
            index.add("id" + std::to_string(i), vectors[i]);
        }

        // Measure search time
        auto start = std::chrono::high_resolution_clock::now();
        auto results = index.search(vectors[0], 10);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::microseconds>(
            end - start
        ).count();

        REQUIRE(duration < 10000); // Should complete within 10ms
    }
}