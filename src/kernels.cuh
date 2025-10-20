#pragma once

#include "kernels/00_interleaved_address_naive.cuh"
#include "kernels/01_interleaved_address_divergence_resolved.cuh"
#include "kernels/02_sequential_address.cuh"
#include "kernels/03_thread_coarsening.cuh"
// #include "kernels/03_halve_block_num.cuh"
// #include "kernels/04_unroll_last_wrap.cuh"
// #include "kernels/05_fully_unroll.cuh"
// #include "kernels/06_warp_shuffle.cuh"
// #include "kernels/07_vectorize_load.cuh"