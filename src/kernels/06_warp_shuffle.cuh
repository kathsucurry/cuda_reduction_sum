#pragma once

#include "../utils.cuh"


template <size_t NUM_THREADS>
__global__ void batched_warp_shuffle(
    float* __restrict__ Y,
    float const* __restrict__ X,
    size_t num_elements_per_batch
) {
    static_assert(NUM_THREADS % 32 == 0, "NUM_THREADS must be a multiple of 32.");

    constexpr size_t NUM_WARPS{NUM_THREADS / 32};
    size_t const block_idx{blockIdx.x};
    size_t const thread_idx{threadIdx.x};

    // The shared data is now among warps rather than threads.
    __shared__ float shared_data[NUM_WARPS];
    
    // Shift the input accordingly to the batch (block) index.
    X += block_idx * num_elements_per_batch;

    // Compute the number of elements each thread will process.
    size_t const num_elements_per_thread{(num_elements_per_batch + NUM_THREADS - 1) / NUM_THREADS};

    // Initialize the sum variable.
    float sum{0.0f};

    // Handle elements of the indices > thread index.
    for (size_t i = 0; i < num_elements_per_thread; ++i) {
        size_t const offset{thread_idx + i * NUM_THREADS};
        if (offset < num_elements_per_batch)
            sum += X[offset];
    }

    constexpr unsigned int FULL_MASK{0xffffffff};
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    // Store the warp element sum in shared_memory.
    if (thread_idx % 32 == 0)
        shared_data[thread_idx / 32] = sum;
    __syncthreads();

    // Determine active threads for obtaining block sum.
    unsigned int const active_threads_mask = __ballot_sync(FULL_MASK, thread_idx < NUM_WARPS);

    if (thread_idx < NUM_WARPS) {
        // Reuse sum variable to store the shared memory elements.
        sum = shared_data[thread_idx];
        for (size_t offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(active_threads_mask, sum, offset);
        }
    }
    
    if (thread_idx == 0)
        Y[block_idx] = sum;
}


template <size_t NUM_THREADS>
void launch_batched_warp_shuffle(
    float* Y,
    float const* X,
    size_t batch_size,
    size_t num_elements_per_batch,
    cudaStream_t stream
) {
    size_t const num_blocks{batch_size};
    batched_warp_shuffle<NUM_THREADS>
        <<<num_blocks, NUM_THREADS, 0, stream>>>(Y, X, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}


template <size_t NUM_THREADS>
void profile_warp_shuffle(
    size_t string_width,
    Elements& elements,
    float* Y_d,
    float *X_d,
    cudaStream_t stream,
    size_t batch_size, size_t num_elements_per_batch
) {
    std::cout << "Batched reduce sum - WARP SHUFFLE" << std::endl;
    profile_batched_kernel(
        launch_batched_warp_shuffle<NUM_THREADS>,
        elements, Y_d, X_d, stream,
        batch_size, num_elements_per_batch
    );
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
