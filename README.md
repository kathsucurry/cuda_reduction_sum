# Setup

```
mkdir build && cd build
cmake ..
cmake --build .
```

# Debugging

The threads in Kernel 0 to 2 only process one element each, i.e., `num_elements_per_batch` and `NUM_THREADS_PER_BATCH` must be identical. If the values differ, it would cause incorrect sum error since each thread only processes one element. Automating or adding additional checks to avoid this issue would be one possible next step.


# Resources

- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf) by Mark Harris
- [CUDA Reduction](https://leimao.github.io/blog/CUDA-Reduction/) by Lei Mao
- [Using CUDA Warp-Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/) by Yuan Lin and Vinod Grover

