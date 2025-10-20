#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

#include "src/elements.h"
#include "src/utils.cuh"
#include "src/kernels.cuh"


void print_gpu_device_info(size_t width) {
    std::cout << std_string_centered("", width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU Device Info", width, ' ') << std::endl;
    std::cout << std_string_centered("", width, '~') << std::endl;

    // Query device name and peak memory bandwidth.
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) / (1 << 30)};
    std::cout << "Global memory size: " << memory_size << " GB" << std::endl;

    float const shared_mem_per_sm{static_cast<float>(device_prop.sharedMemPerMultiprocessor) / 1024.0f};
    std::cout << "Shared memory size per SM: " << shared_mem_per_sm << " KB" << std::endl;

    float const shared_mem_per_block{static_cast<float>(device_prop.sharedMemPerBlock) / 1024.0f};
    std::cout << "Shared memory size per block: " << shared_mem_per_block << " KB" << std::endl;

    int const max_threads_per_sm{device_prop.maxThreadsPerMultiProcessor};
    std::cout << "Max threads per SM: " << max_threads_per_sm << std::endl;

    // Calculate peak bandwidth.
    // 1) Obtain memory clock rate in kHz and convert to Hz.
    double const memory_clock_hz{device_prop.memoryClockRate * 1000.0};
    // 2) Obtain memory bus width in bits and convert to bytes.
    double const memory_bus_width_bytes{device_prop.memoryBusWidth / 8.0};
    std::cout << "Memory bus width: " << memory_bus_width_bytes << " bytes" << std::endl;
    // 3) Factor of 2.0 for Dual Data Rate (DDR), then divide by 1.0e9 to convert from bytes/second to GB/second.
    float const peak_bandwidth{static_cast<float>(2.0f * memory_clock_hz * memory_bus_width_bytes / 1.0e9)};
    std::cout << "Peak bandwidth: " << peak_bandwidth << " GB/s" << std::endl;

    std::cout << std_string_centered("", width, '~') << std::endl;
}


void print_profiling_header(size_t width, size_t const batch_size, size_t const num_elements_per_batch) {
    std::cout << std_string_centered("", width, '~') << std::endl;
    std::cout << std_string_centered("Reduce Sum Profiling", width, ' ')
              << std::endl;
    std::cout << std_string_centered("", width, '~') << std::endl;

    std::cout << std_string_centered("", width, '=') << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Number of Elements Per Batch: " << num_elements_per_batch
              << std::endl;
    std::cout << "Total number of elements " << batch_size * num_elements_per_batch << std::endl;
    std::cout << std_string_centered("", width, '=') << std::endl;
}


int main() {
    // Print the GPU device info.
    constexpr size_t string_width{50U};
    print_gpu_device_info(string_width);

    // Batch here represents blocks, i.e., batch_size = number of blocks.
    // When thread coarsening is used, make sure that `num_elements_per_batch` =
    // `NUM_THREADS_PER_BATCH` * # elements per thread.
    size_t const batch_size{2048 * 2048};
    size_t const num_elements_per_batch{128 * 1};
    print_profiling_header(string_width, batch_size, num_elements_per_batch);

    constexpr size_t NUM_THREADS_PER_BATCH{128};
    static_assert(NUM_THREADS_PER_BATCH % 32 == 0, "NUM_THREADS_PER_BATCH must be a multiple of 32.");

    size_t const num_elements{batch_size * num_elements_per_batch};

    // Prepare stream and allocate device memory.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    // Generate list elements.
    // RandomElements elements(num_elements, batch_size, num_elements_per_batch);
    ConstantElements elements(num_elements, batch_size, num_elements_per_batch);

    float* X_d;
    float *Y_d;

    CHECK_CUDA_ERROR(cudaMalloc(&X_d, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&Y_d, batch_size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(X_d, elements.X.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    // profile_interleaved_address_naive<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     elements,
    //     Y_d, X_d,
    //     stream,
    //     batch_size, num_elements_per_batch);

    // profile_interleaved_address_divergence_resolved<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     elements,
    //     Y_d, X_d,
    //     stream,
    //     batch_size, num_elements_per_batch);

    profile_sequential_address<NUM_THREADS_PER_BATCH>(
        string_width,
        elements,
        Y_d, X_d,
        stream,
        batch_size, num_elements_per_batch);

    profile_thread_coarsening<NUM_THREADS_PER_BATCH>(
        string_width,
        elements,
        Y_d, X_d,
        stream,
        batch_size, num_elements_per_batch);
    
    // // // profile_halve_block_num<NUM_THREADS_PER_BATCH>(
    // // //     string_width,
    // // //     Y,
    // // //     Y_d, X_d,
    // // //     stream,
    // // //     element_value,
    // // //     batch_size, num_elements_per_batch);

    // // profile_unroll_last_wrap<NUM_THREADS_PER_BATCH>(
    // //     string_width,
    // //     Y,
    // //     Y_d, X_d,
    // //     stream,
    // //     element_value,
    // //     batch_size, num_elements_per_batch);

    // // profile_fully_unroll<NUM_THREADS_PER_BATCH>(
    // //     string_width,
    // //     Y,
    // //     Y_d, X_d,
    // //     stream,
    // //     element_value,
    // //     batch_size, num_elements_per_batch);

    // // profile_warp_shuffle<NUM_THREADS_PER_BATCH>(
    // //     string_width,
    // //     Y,
    // //     Y_d, X_d,
    // //     stream,
    // //     element_value,
    // //     batch_size, num_elements_per_batch);

    // // profile_vectorize_load<NUM_THREADS_PER_BATCH>(
    // //     string_width,
    // //     Y,
    // //     Y_d, X_d,
    // //     stream,
    // //     element_value,
    // //     batch_size, num_elements_per_batch);


    CHECK_CUDA_ERROR(cudaFree(X_d));
    CHECK_CUDA_ERROR(cudaFree(Y_d));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    return 0;
}

