#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <string>

#include "elements.h"


#define CHECK_CUDA_ERROR(value) check((value), #value, __FILE__, __LINE__)
void check(cudaError_t error, char const* func, char const* file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA runtime error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(error) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const* file, int line) {
    cudaError_t const error{cudaGetLastError()};
    if (error != cudaSuccess) {
        std::cerr << "CUDA runtime error at: " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}


std::string std_string_centered(std::string const& str, size_t width, char padding = ' ') {
    size_t const length{str.length()};
    
    if (width < length)
        throw std::runtime_error("Width is too small.");
    
    size_t const left_padding{(width - length) / 2};
    size_t const right_padding{width - length - left_padding};
    std::string const str_centered{std::string(left_padding, padding) + str +
                                   std::string(right_padding, padding)};
    return str_centered;
}


template <class T>
float measure_performance_in_ms(
    std::function<T(cudaStream_t)> bound_function,
    cudaStream_t stream, size_t num_repeats = 50,
    size_t num_warmups = 10
) {
    cudaEvent_t start, stop;
    float elapsed_time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i = 0; i < num_warmups; ++i)
        bound_function(stream);
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i = 0; i < num_repeats; ++i)
        bound_function(stream);
    
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency_ms{elapsed_time / num_repeats};

    return latency_ms;
}


/**
 * Profiles the kernel.
 * 
 * It does the following 3 steps:
 * 1) Invoke kernel once for verification purposes.
 * 2) Measure performance by running warmups and repeated computations.
 * 
 * Since reduction has a very low arithmetic intensity (i.e., memory-bound), we're mostly
 * interested in the bandwidth.
 */
void profile_batched_kernel(
    std::function<void(float*, float const*, size_t, size_t, cudaStream_t)> batched_launch_function,
    Elements& elements,
    float* Y_d,
    float *X_d,
    cudaStream_t stream,
    size_t batch_size, size_t num_elements_per_batch
) {
    size_t const num_elements{batch_size * num_elements_per_batch};

    batched_launch_function(Y_d, X_d, batch_size, num_elements_per_batch, stream);
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Verify the correctness of the kernel.
    CHECK_CUDA_ERROR(cudaMemcpy(elements.Y.data(), Y_d, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    elements.verify_kernel();
    
    std::function<void(cudaStream_t)> const bound_function{
        std::bind(
            batched_launch_function, Y_d, X_d, batch_size, num_elements_per_batch, std::placeholders::_1
        )
    };
    float const latency{measure_performance_in_ms<void>(bound_function, stream)};
    std::cout << "Latency: " << latency * 1000.0 << " us" << std::endl;

    // Compute effective bandwidth.
    size_t num_bytes{num_elements * sizeof(float) + batch_size * sizeof(float)};
    float const bandwidth{(num_bytes * 1e-6f) / latency};
    std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;
}
