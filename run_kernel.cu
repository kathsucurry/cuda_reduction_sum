#include <cuda_runtime.h>
#include <iostream>
#include <string>

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
    std::cout << "Memory size: " << memory_size << " GB" << std::endl;

    // Calculate peak bandwidth.
    // 1) Obtain memory clock rate in kHz and convert to Hz.
    double const memory_clock_hz{device_prop.memoryClockRate * 1000.0};
    // 2) Obtain memory bus width in bits and convert to bytes.
    double const memory_bus_width_bytes{device_prop.memoryBusWidth / 8.0};
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
    std::cout << std_string_centered("", width, '=') << std::endl;
}


int main() {
    // Print the GPU device info.
    constexpr size_t string_width{50U};
    print_gpu_device_info(string_width);

    size_t const batch_size{2048 * 256};
    size_t const num_elements_per_batch{1024};
    print_profiling_header(string_width, batch_size, num_elements_per_batch);

    constexpr size_t NUM_THREADS_PER_BATCH{1024};
    static_assert(NUM_THREADS_PER_BATCH % 32 == 0, "NUM_THREADS_PER_BATCH must be a multiple of 32.");
    static_assert(NUM_THREADS_PER_BATCH <= 1024, "NUM_THREADS_PER_BATCH must be <= 1024.");

    size_t const num_elements{batch_size * num_elements_per_batch};

    // Prepare stream and allocate device memory.
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    constexpr float element_value{1.0f};
    std::vector<float> X(num_elements, element_value);
    std::vector<float> Y(batch_size, 0.0f);

    float* X_d;
    float *Y_d;

    CHECK_CUDA_ERROR(cudaMalloc(&X_d, num_elements * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&Y_d, batch_size * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(X_d, X.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice));

    profile_naive<NUM_THREADS_PER_BATCH>(
        string_width,
        Y,
        Y_d, X_d,
        stream,
        element_value,
        batch_size, num_elements_per_batch);
    
    // profile_interleaved_address_1<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     Y,
    //     Y_d, X_d,
    //     stream,
    //     element_value,
    //     batch_size, num_elements_per_batch);
    
    // profile_interleaved_address_2<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     Y,
    //     Y_d, X_d,
    //     stream,
    //     element_value,
    //     batch_size, num_elements_per_batch);
    
    // profile_sequential_address<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     Y,
    //     Y_d, X_d,
    //     stream,
    //     element_value,
    //     batch_size, num_elements_per_batch);
    
    // // profile_halve_block_num<NUM_THREADS_PER_BATCH>(
    // //     string_width,
    // //     Y,
    // //     Y_d, X_d,
    // //     stream,
    // //     element_value,
    // //     batch_size, num_elements_per_batch);

    // profile_unroll_last_wrap<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     Y,
    //     Y_d, X_d,
    //     stream,
    //     element_value,
    //     batch_size, num_elements_per_batch);

    // profile_fully_unroll<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     Y,
    //     Y_d, X_d,
    //     stream,
    //     element_value,
    //     batch_size, num_elements_per_batch);

    // profile_warp_shuffle<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     Y,
    //     Y_d, X_d,
    //     stream,
    //     element_value,
    //     batch_size, num_elements_per_batch);

    // profile_vectorize_load<NUM_THREADS_PER_BATCH>(
    //     string_width,
    //     Y,
    //     Y_d, X_d,
    //     stream,
    //     element_value,
    //     batch_size, num_elements_per_batch);


    CHECK_CUDA_ERROR(cudaFree(X_d));
    CHECK_CUDA_ERROR(cudaFree(Y_d));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    
    return 0;
}

