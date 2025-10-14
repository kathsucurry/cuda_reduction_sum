#pragma once

#include "../utils.cuh"


template <size_t NUM_THREADS>
__device__ void warp_reduce(volatile float* shared_data, size_t thread_idx) {
    if (NUM_THREADS >= 64) shared_data[thread_idx] += shared_data[thread_idx + 32];
    if (NUM_THREADS >= 32) shared_data[thread_idx] += shared_data[thread_idx + 16];
    if (NUM_THREADS >= 16) shared_data[thread_idx] += shared_data[thread_idx + 8];
    if (NUM_THREADS >= 8) shared_data[thread_idx] += shared_data[thread_idx + 4];
    if (NUM_THREADS >= 4) shared_data[thread_idx] += shared_data[thread_idx + 2];
    if (NUM_THREADS >= 2) shared_data[thread_idx] += shared_data[thread_idx + 1];
}


template <size_t NUM_THREADS, size_t NUM_THREADS_PER_WRAP>
__device__ float block_fully_unroll(
    float shared_data[NUM_THREADS],
    float const* __restrict__ X,
    size_t num_elements
) {
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32.");
    size_t const num_elements_per_thread{(num_elements + NUM_THREADS - 1) / NUM_THREADS};

    size_t const thread_idx{threadIdx.x};
    float sum{0.0f};

    // Handle elements of the indices > thread index.
    for (size_t i = 0; i < num_elements_per_thread; ++i) {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements)
            sum += X[offset];
    }
    shared_data[thread_idx] = sum;
    __syncthreads();

    if (NUM_THREADS == 1024){
        if (thread_idx < 512) shared_data[thread_idx] += shared_data[thread_idx + 512];
        __syncthreads();
    }
    if (NUM_THREADS >= 512){
        if (thread_idx < 256) shared_data[thread_idx] += shared_data[thread_idx + 256];
        __syncthreads();
    }
    if (NUM_THREADS >= 256){
        if (thread_idx < 128) shared_data[thread_idx] += shared_data[thread_idx + 128];
        __syncthreads();
    }
    if (NUM_THREADS >= 128){
        if (thread_idx < 64) shared_data[thread_idx] += shared_data[thread_idx + 64];
        __syncthreads();
    }

    if (thread_idx < NUM_THREADS_PER_WRAP)
        warp_reduce<NUM_THREADS>(shared_data, thread_idx);

    return shared_data[0];
}


template <size_t NUM_THREADS, size_t NUM_THREADS_PER_WRAP>
__global__ void batched_fully_unroll(
    float* __restrict__ Y,
    float const* __restrict__ X,
    size_t num_elements_per_batch
) {
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32.");

    size_t const block_idx{blockIdx.x};
    size_t const thread_idx{threadIdx.x};
    __shared__ float shared_data[NUM_THREADS];
    size_t const num_elements_per_thread{(num_elements_per_batch + NUM_THREADS - 1) / NUM_THREADS};
    
    X += block_idx * num_elements_per_batch;
    float sum{0.0f};

    // Handle elements of the indices > thread index.
    for (size_t i = 0; i < num_elements_per_thread; ++i) {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements_per_batch)
            sum += X[offset];
    }
    shared_data[thread_idx] = sum;
    __syncthreads();

    if (NUM_THREADS == 1024){
        if (thread_idx < 512) shared_data[thread_idx] += shared_data[thread_idx + 512];
        __syncthreads();
    }
    if (NUM_THREADS >= 512){
        if (thread_idx < 256) shared_data[thread_idx] += shared_data[thread_idx + 256];
        __syncthreads();
    }
    if (NUM_THREADS >= 256){
        if (thread_idx < 128) shared_data[thread_idx] += shared_data[thread_idx + 128];
        __syncthreads();
    }
    if (NUM_THREADS >= 128){
        if (thread_idx < 64) shared_data[thread_idx] += shared_data[thread_idx + 64];
        __syncthreads();
    }

    if (thread_idx < NUM_THREADS_PER_WRAP)
        warp_reduce<NUM_THREADS>(shared_data, thread_idx);

    if (thread_idx == 0)
        Y[block_idx] = shared_data[0];
}


template <size_t NUM_THREADS>
void launch_batched_fully_unroll(
    float* Y,
    float const* X,
    size_t batch_size,
    size_t num_elements_per_batch,
    cudaStream_t stream
) {
    constexpr size_t NUM_THREADS_PER_WARP{32};
    size_t const num_blocks{batch_size};
    batched_fully_unroll<NUM_THREADS, NUM_THREADS_PER_WARP>
        <<<num_blocks, NUM_THREADS, 0, stream>>>(Y, X, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}


template <size_t NUM_THREADS>
void profile_fully_unroll(
    size_t string_width,
    std::vector<float> Y,
    float* Y_d,
    float *X_d,
    cudaStream_t stream,
    float element_value,
    size_t batch_size, size_t num_elements_per_batch
) {
    std::cout << "Batched reduce sum - FULL UNROLL" << std::endl;
    profile_batched_kernel(
        launch_batched_fully_unroll<NUM_THREADS>,
        Y, Y_d, X_d, stream, element_value,
        batch_size, num_elements_per_batch
    );
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
