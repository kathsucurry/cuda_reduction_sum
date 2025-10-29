# Reduction (Sum)

This code repo corresponds to the "Reduction (Sum)" blog post series:

- [Part 1: introduction](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html)
- [Part 2: implementation](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html)

## Setup

```
mkdir build && cd build
cmake ..
cmake --build .
```

## Notes

The threads in Kernel 0 to 3 only process one element each, i.e., **`num_elements_per_batch` and `NUM_THREADS_PER_BATCH` must be identical**. If the values differ, it would cause incorrect sum error since each thread only processes one element.

Similarly, `num_elements_per_batch` for running Kernel 7 has to be `NUM_THREADS_PER_BATCH * <multiple of 4>`.

A next step for cleaner code and easier usage is to automate or add additional checks to prevent the issues above.


# Resources

- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) by Mark Harris (2007)
- [CUDA Reduction](https://leimao.github.io/blog/CUDA-Reduction/) by Lei Mao (2024)
- [Programming Massively Parallel Processors, 4th Edition](https://www.amazon.com/Programming-Massively-Parallel-Processors-Hands/dp/0323912311) by Hwu, Kirk, and Hajj (2023)
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/) by Justin Luitjens (2014)
- [CUDA Pro Tip: Do The Kepler Shuffler](https://developer.nvidia.com/blog/cuda-pro-tip-kepler-shuffle/) by Mark Harris (2014)
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) by Yuan Lin and Vinod Grover (2018)