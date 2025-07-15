#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <algorithm>
#include <cfloat>

#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

constexpr int TILE_SIZE = 512;

/**
 * CUDA kernel for bitonic sort using shared memory
 */
__global__ void sortTiles(float* data, int n) {
  __shared__ float shared_data[TILE_SIZE];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int tile_start = bid * TILE_SIZE;

  // Load tile data into shared memory
  shared_data[tid] = data[tile_start + tid];
  __syncthreads();

  // Sort the tile using bitonic sort
  for (int k = 2; k <= TILE_SIZE; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int partner = tid ^ j;
      if (partner > tid) {
        bool ascending = (tid & k) == 0;
        if ((shared_data[tid] > shared_data[partner]) == ascending) {
          // Swap
          float temp = shared_data[tid];
          shared_data[tid] = shared_data[partner];
          shared_data[partner] = temp;
        }
      }
      __syncthreads();
    }
  }

  // Write sorted tile back to global memory
  if (tile_start + tid < n) {
    data[tile_start + tid] = shared_data[tid];
  }
}

/**
 * Kernel for global bitonic merge step
 */
__global__ void bitonicMergeStep(float* data, int n, int j, int k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int partner = tid ^ j;

  if (partner > tid && tid < n && partner < n) {
    bool ascending = (tid & k) == 0;
    if ((data[tid] > data[partner]) == ascending) {
      float temp = data[tid];
      data[tid] = data[partner];
      data[partner] = temp;
    }
  }
}

/**
 * Host function for tiled bitonic sort using shared memory
 */
extern "C" void bitonicSortSharedMemory(float* h_data, int n) {
  float* d_data;
  size_t size = n * sizeof(float);

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_data, size));

  // Copy to device
  CUDA_CHECK(
      cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));

  // Multiple tiles - use tiled approach
  int num_tiles = n / TILE_SIZE;

  // Sort individual tiles using shared memory
  int shared_mem_size = TILE_SIZE * sizeof(float);
  sortTiles<<<num_tiles, TILE_SIZE, shared_mem_size>>>(d_data, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Use complete bitonic sort to merge all tiles
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  // Apply complete bitonic sort starting from the beginning
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonicMergeStep<<<grid_size, block_size>>>(d_data, n, j, k);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_data));
}

/**
 * CPU implementation for comparison
 */
extern "C" void bitonicSortCPU(float* data, int n) {
  std::sort(data, data + n);
}
