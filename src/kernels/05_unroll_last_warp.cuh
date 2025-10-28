#pragma once

#include "../utils.cuh"


__device__ void warp_reduce(volatile float* shared_data, size_t thread_idx) {
    shared_data[thread_idx] += shared_data[thread_idx + 32];
    shared_data[thread_idx] += shared_data[thread_idx + 16];
    shared_data[thread_idx] += shared_data[thread_idx + 8];
    shared_data[thread_idx] += shared_data[thread_idx + 4];
    shared_data[thread_idx] += shared_data[thread_idx + 2];
    shared_data[thread_idx] += shared_data[thread_idx + 1];
}


template <size_t NUM_THREADS, size_t NUM_THREADS_PER_WARP>
__global__ void batched_unroll_last_warp(
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

    // Compute the number of elements each thread will process.
    size_t const num_elements_per_thread{(num_elements_per_batch + NUM_THREADS - 1) / NUM_THREADS};
    
    // Initialize the sum variable.
    float sum{0.0f};
    
    for (size_t i = 0; i < num_elements_per_thread; ++i) {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements_per_batch)
            sum += X[offset];
    }
    shared_data[thread_idx] = sum;

    for (size_t stride = NUM_THREADS / 2; stride > NUM_THREADS_PER_WARP; stride >>= 1) {
        __syncthreads();
        if (thread_idx < stride)
            shared_data[thread_idx] += shared_data[thread_idx + stride];
    }

    if (thread_idx < NUM_THREADS_PER_WARP) {
        __syncthreads();
        warp_reduce(shared_data, thread_idx);
    }

    if (thread_idx == 0)
        Y[block_idx] = shared_data[0];
}


template <size_t NUM_THREADS>
void launch_batched_unroll_last_warp(
    float* Y,
    float const* X,
    size_t batch_size,
    size_t num_elements_per_batch,
    cudaStream_t stream
) {
    constexpr size_t NUM_THREADS_PER_WARP{32};
    size_t const num_blocks{batch_size};
    batched_unroll_last_warp<NUM_THREADS, NUM_THREADS_PER_WARP>
        <<<num_blocks, NUM_THREADS, 0, stream>>>(Y, X, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}


template <size_t NUM_THREADS>
void profile_unroll_last_warp(
    size_t string_width,
    Elements& elements,
    float* Y_d,
    float *X_d,
    cudaStream_t stream,
    size_t batch_size, size_t num_elements_per_batch
) {
    std::cout << "Batched reduce sum - UNROLL LAST WARP" << std::endl;
    profile_batched_kernel(
        launch_batched_unroll_last_warp<NUM_THREADS>,
        elements, Y_d, X_d, stream,
        batch_size, num_elements_per_batch
    );
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
