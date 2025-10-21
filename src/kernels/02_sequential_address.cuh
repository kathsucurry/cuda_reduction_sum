#pragma once

#include "../utils.cuh"
#include "../elements.h"


template <size_t NUM_THREADS>
__global__ void batched_sequential_address(
    float* __restrict__ Y,
    float const* __restrict__ X,
    size_t num_elements_per_batch
) {
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32.");

    size_t const block_idx{blockIdx.x};
    size_t const thread_idx{threadIdx.x};
    __shared__ float shared_data[NUM_THREADS];

    // Shift the input accordingly to the batch (block) index.
    X += block_idx * num_elements_per_batch;
    // Store a single element per thread in shared memory.
    shared_data[thread_idx] = X[thread_idx];
    __syncthreads();

    for (size_t stride = NUM_THREADS / 2; stride > 0; stride >>= 1) {
        if (thread_idx < stride)
            shared_data[thread_idx] += shared_data[thread_idx + stride];
        __syncthreads();
    }

    if (thread_idx == 0)
        Y[block_idx] = shared_data[0];
}


template <size_t NUM_THREADS>
void launch_batched_sequential_address(
    float* Y,
    float const* X,
    size_t batch_size,
    size_t num_elements_per_batch,
    cudaStream_t stream
) {
    size_t const num_blocks{batch_size};
    batched_sequential_address<NUM_THREADS>
        <<<num_blocks, NUM_THREADS, 0, stream>>>(Y, X, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}


template <size_t NUM_THREADS>
void profile_sequential_address(
    size_t string_width,
    Elements& elements,
    float* Y_d,
    float *X_d,
    cudaStream_t stream,
    size_t batch_size, size_t num_elements_per_batch
) {
    std::cout << "Batched reduce sum - SEQUENTIAL ADDRESS" << std::endl;
    profile_batched_kernel(
        launch_batched_sequential_address<NUM_THREADS>,
        elements, Y_d, X_d, stream,
        batch_size, num_elements_per_batch
    );
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
