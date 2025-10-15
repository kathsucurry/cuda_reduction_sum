#pragma once

#include <algorithm>
#include <vector>
#include <random>

#define EPS 1e-3


class Elements {
public:
    std::vector<float> X;
    std::vector<float> Y;
    size_t num_elements;
    size_t num_elements_per_batch;
    size_t batch_size;

    Elements(size_t num_elements, size_t batch_size, size_t num_elements_per_batch)
    : X(num_elements),
      Y(batch_size, 0.0f),
      num_elements(num_elements),
      batch_size(batch_size),
      num_elements_per_batch(num_elements_per_batch) {}

    virtual void verify_kernel() const = 0;
};


class RandomElements : public Elements {
    public:
        RandomElements(size_t num_elements, size_t batch_size, size_t num_elements_per_batch)
        : Elements(num_elements, batch_size, num_elements_per_batch) {
            // Initialize X with random floating-point elements.
            std::random_device rd;
            std::mt19937 generator(rd());
            std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
            std::generate(X.begin(), X.end(), [&]() { return distribution(generator); });
        }

        void verify_kernel() const override {
            for (size_t i = 0; i < batch_size; ++i) {
                // Compute the partial sum in CPU.
                float expected_sum{
                    std::reduce(
                        X.begin() + i * num_elements_per_batch,
                        X.begin() + (i + 1) * num_elements_per_batch
                    )};
        
                if (abs(Y.at(i) - expected_sum) > EPS) {
                    std::cout << "Expected: " << expected_sum
                              << " but got: " << Y.at(i) << std::endl;
                    throw std::runtime_error("Error: incorrect sum");
                }
            }
        }
};


class ConstantElements : public Elements {
    public:
        float element_value;

        ConstantElements(size_t num_elements, size_t batch_size, size_t num_elements_per_batch, float const value = 1.0f)
        : Elements(num_elements, batch_size, num_elements_per_batch),
          element_value(value) {
            std::fill(X.begin(), X.end(), element_value);
          }

        void verify_kernel() const override {
            for (size_t i = 0; i < batch_size; ++i) {
                if (Y.at(i) != num_elements_per_batch * element_value) {
                    std::cout << "Expected: " << num_elements_per_batch * element_value
                              << " but got: " << Y.at(i) << std::endl;
                    throw std::runtime_error("Error: incorrect sum");
                }
            }
        }
};