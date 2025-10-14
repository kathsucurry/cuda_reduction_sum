#pragma once

#include "../utils.cuh"


template <size_t NUM_THREADS>
__global__ void batched_vectorize_load(
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

    X += block_idx * num_elements_per_batch;
    size_t const num_elements_per_thread{(num_elements_per_batch + NUM_THREADS - 1) / NUM_THREADS};
    float sum{0.0f};

    // Handle elements of the indices > thread index.
    for (size_t i = 0; i < num_elements_per_thread / 4; ++i) {
        size_t const offset{4 * (thread_idx + i * NUM_THREADS)};
        if (offset < num_elements_per_batch) {
            float4 const tmp = reinterpret_cast<float4 const*>(&X[offset])[0];
            sum += tmp.x + tmp.y + tmp.z + tmp.w;
        }
    }

    constexpr unsigned int FULL_MASK{0xffffffff};
#pragma unroll
    for (size_t offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(FULL_MASK, sum, offset);
    }

    if (thread_idx % 32 == 0)
        shared_data[thread_idx / 32] = sum;
    __syncthreads();

    float block_sum{0.0f};
#pragma unroll
    for (size_t i = 0; i < NUM_WARPS; ++i)
        block_sum += shared_data[i];
    
    if (thread_idx == 0)
        Y[block_idx] = block_sum;
}


template <size_t NUM_THREADS>
void launch_batched_vectorize_load(
    float* Y,
    float const* X,
    size_t batch_size,
    size_t num_elements_per_batch,
    cudaStream_t stream
) {
    size_t const num_blocks{batch_size};
    batched_vectorize_load<NUM_THREADS>
        <<<num_blocks, NUM_THREADS, 0, stream>>>(Y, X, num_elements_per_batch);
    CHECK_LAST_CUDA_ERROR();
}


template <size_t NUM_THREADS>
void profile_vectorize_load(
    size_t string_width,
    std::vector<float> Y,
    float* Y_d,
    float *X_d,
    cudaStream_t stream,
    float element_value,
    size_t batch_size, size_t num_elements_per_batch
) {
    std::cout << "Batched reduce sum - VECTORIZE LOAD" << std::endl;
    profile_batched_kernel(
        launch_batched_vectorize_load<NUM_THREADS>,
        Y, Y_d, X_d, stream, element_value,
        batch_size, num_elements_per_batch
    );
    std::cout << std_string_centered("", string_width, '-') << std::endl;
}
