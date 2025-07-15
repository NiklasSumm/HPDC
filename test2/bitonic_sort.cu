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

template <int TILE_SIZE>
__global__ void BitonicSort_shared(float* data) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float shared_data[TILE_SIZE];

  shared_data[threadIdx.x] = data[tid];

  __syncthreads();

  for (int k = 2; k <= TILE_SIZE; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      int partner = threadIdx.x ^ j;
      if (partner > threadIdx.x && partner < TILE_SIZE) {
        bool asc = ((threadIdx.x & k) == 0);
        if ((shared_data[threadIdx.x] > shared_data[partner]) == asc) {
          float tmp = shared_data[threadIdx.x];
          shared_data[threadIdx.x] = shared_data[partner];
          shared_data[partner] = tmp;
        }
      }
      __syncthreads();
    }
  }

  data[tid] = shared_data[threadIdx.x];
}

template <int TILE_SIZE>
__global__ void BitonicSteps_shared(float* data, int k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float shared_data[TILE_SIZE];

  shared_data[threadIdx.x] = data[tid];

  __syncthreads();

  for (int j = (TILE_SIZE >> 1); j > 0; j >>= 1) {
    int partner = threadIdx.x ^ j;
    if (partner > threadIdx.x && partner < TILE_SIZE) {
      bool asc = ((tid & k) == 0);
      if ((shared_data[threadIdx.x] > shared_data[partner]) == asc) {
        float tmp = shared_data[threadIdx.x];
        shared_data[threadIdx.x] = shared_data[partner];
        shared_data[partner] = tmp;
      }
    }
    __syncthreads();
  }

  data[tid] = shared_data[threadIdx.x];
}

template <int N>
struct BitonicSortConfig {
  static_assert((N & (N - 1)) == 0, "N must be power of two");
  static constexpr int TILE_SIZE = 1024;
  static constexpr int BLOCK_SIZE = TILE_SIZE;
  static constexpr int NUM_TILES = N / TILE_SIZE;
  static constexpr int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  static constexpr int SHARED_MEM = TILE_SIZE * sizeof(float);
};

__global__ void BitonicStep(float* data, int n, int j, int k) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int partner = tid ^ j;
  if (partner > tid && partner < n) {
    bool asc = ((tid & k) == 0);
    if ((data[tid] > data[partner]) == asc) {
      float tmp = data[tid];
      data[tid] = data[partner];
      data[partner] = tmp;
    }
  }
}

template <int N>
void sort_fixed_size(float* d_data) {
  using Config = BitonicSortConfig<N>;

  constexpr int TILE_SIZE = Config::TILE_SIZE;
  constexpr int BLOCK_SIZE = Config::BLOCK_SIZE;
  constexpr int NUM_TILES = Config::NUM_TILES;
  constexpr int GRID_SIZE = Config::GRID_SIZE;
  constexpr int SHARED_MEM = Config::SHARED_MEM;

  // Since bitonic sort relys on comparisons between near and distant entries,
  // we would need to fit the entire array into shared memory to compute it in a
  // single kernel however this is not possible for larger sizes there for we
  // chose the following approch utilizing 3 different kernels:
  //  -the first kernel executes the first bitonic steps to sort one tile per
  //  block using shared memory -the second kernel is called iteratively to
  //  execute the bitonic steps where entries from different tiles need to be
  //  compared -the third kernel is called when comparison again happens only
  //  inside the tiles. It executes multiple steps and uses shared memory

  // Sort each tile with bitonic sort using shared memory
  BitonicSort_shared<TILE_SIZE><<<NUM_TILES, BLOCK_SIZE, SHARED_MEM>>>(d_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int k = TILE_SIZE; k <= N; k <<= 1) {
    for (int j = k >> 1; j >= TILE_SIZE; j >>= 1) {
      // execute single bitonic sort step directly on global memory (so no
      // shared memory usage here)
      BitonicStep<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, N, j, k);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // executes the bitonic steps for j = TILE_SIZE/2 down to j = 1 using shared
    // memory
    BitonicSteps_shared<TILE_SIZE><<<NUM_TILES, TILE_SIZE>>>(d_data, k);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void sort_fallback(float* d_data, int n) {
  constexpr int TILE_SIZE = 1024;
  int BLOCK_SIZE = TILE_SIZE;
  int NUM_TILES = n / TILE_SIZE;
  int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int SHARED_MEM = TILE_SIZE * sizeof(float);

  // Since bitonic sort relys on comparisons between near and distant entries,
  // we would need to fit the entire array into shared memory to compute it in a
  // single kernel however this is not possible for larger sizes there for we
  // chose the following approch utilizing 3 different kernels:
  //  -the first kernel executes the first bitonic steps to sort one tile per
  //  block using shared memory -the second kernel is called iteratively to
  //  execute the bitonic steps where entries from different tiles need to be
  //  compared -the third kernel is called when comparison again happens only
  //  inside the tiles. It executes multiple steps and uses shared memory

  // Sort each tile with bitonic sort using shared memory
  BitonicSort_shared<TILE_SIZE><<<NUM_TILES, BLOCK_SIZE, SHARED_MEM>>>(d_data);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int k = TILE_SIZE; k <= n; k <<= 1) {
    for (int j = k >> 1; j >= TILE_SIZE; j >>= 1) {
      // execute single bitonic sort step directly on global memory (so no
      // shared memory usage here)
      BitonicStep<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, n, j, k);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // executes the bitonic steps for j = TILE_SIZE/2 down to j = 1 using shared
    // memory
    BitonicSteps_shared<TILE_SIZE><<<NUM_TILES, TILE_SIZE>>>(d_data, k);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

extern "C" void bitonicSortSharedMemory(float* h_data, int n) {
  float* d_data;
  size_t bytes = n * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

  switch (n) {
    case 1024:
      sort_fixed_size<1024>(d_data);
      break;
    case 2048:
      sort_fixed_size<2048>(d_data);
      break;
    case 4096:
      sort_fixed_size<4096>(d_data);
      break;
    case 8192:
      sort_fixed_size<8192>(d_data);
      break;
    case 16384:
      sort_fixed_size<16384>(d_data);
      break;
    case 32768:
      sort_fixed_size<32768>(d_data);
      break;
    case 65536:
      sort_fixed_size<65536>(d_data);
      break;
    case 131072:
      sort_fixed_size<131072>(d_data);
      break;
    case 262144:
      sort_fixed_size<262144>(d_data);
      break;
    case 524288:
      sort_fixed_size<524288>(d_data);
      break;
    case 1048576:
      sort_fixed_size<1048576>(d_data);
      break;
    case 2097152:
      sort_fixed_size<2097152>(d_data);
      break;
    case 4194304:
      sort_fixed_size<4194304>(d_data);
      break;
    case 8388608:
      sort_fixed_size<8388608>(d_data);
      break;
    case 16777216:
      sort_fixed_size<16777216>(d_data);
      break;
    case 33554432:
      sort_fixed_size<33554432>(d_data);
      break;
    default:
      if (n < 1024){
        fprintf(stderr, "Unsupported size %d - array size is smaller than tile size\n", n);
        cudaFree(d_data);
        std::exit(EXIT_FAILURE);
      }
      sort_fallback(d_data, n);
  }

  CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_data));
}

extern "C" void stdSortCPU(float* data, int n) {
  std::sort(data, data + n);
}
