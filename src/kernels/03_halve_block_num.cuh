#pragma once

#include "../utils.cuh"
#include "02_sequential_address.cuh"


template <size_t NUM_THREADS>
void profile_halve_block_num(
    size_t string_width,
    std::vector<float> Y,
    float* Y_d,
    float *X_d,
    cudaStream_t stream,
    float element_value,
    size_t batch_size, size_t num_elements_per_batch
) {
    std::cout << "Batched reduce sum - HALVE BLOCK NUM" << std::endl;
    profile_batched_kernel(
        launch_batched_sequential_address<NUM_THREADS * 2>,
        Y, Y_d, X_d, stream, element_value,
        batch_size, num_elements_per_batch
    );
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
