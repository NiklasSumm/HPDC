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

template <int K>
__device__ void bitonicStep(float* sdata, int tid) {
  constexpr int j = K >> 1;
  int partner = tid ^ j;
  if (partner > tid) {
    bool ascending = (K & tid) == 0;
    if ((sdata[tid] > sdata[partner]) == ascending) {
      // Swap
      float temp = sdata[tid];
      sdata[tid] = sdata[partner];
      sdata[partner] = temp;
    }
  }
}

template <int TILE_SIZE, int K = 2>
__device__ void bitonicSortUnrolled(float* sdata, int tid) {
  if constexpr (K <= TILE_SIZE) {
#pragma unroll
    for (int j = K >> 1; j > 0; j >>= 1) {
      bitonicStep<K>(sdata, tid);
      __syncthreads();
    }
    bitonicSortUnrolled<TILE_SIZE, (K << 1)>(sdata, tid);
  }
}

template <int TILE_SIZE>
__global__ void sortTilesUnrolledKernel(float* data, int n) {
  __shared__ float shared_data[TILE_SIZE];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int tile_start = bid * TILE_SIZE;

  if (tile_start + tid < n) {
    shared_data[tid] = data[tile_start + tid];
  } else {
    shared_data[tid] = FLT_MAX;
  }
  __syncthreads();

  bitonicSortUnrolled<TILE_SIZE>(shared_data, tid);

  if (tile_start + tid < n) {
    data[tile_start + tid] = shared_data[tid];
  }
}

template <int N>
struct BitonicSortConfig {
  static_assert((N & (N - 1)) == 0, "N must be power of two");
  static constexpr int TILE_SIZE = 512;
  static constexpr int BLOCK_SIZE = TILE_SIZE;
  static constexpr int NUM_TILES = N / TILE_SIZE;
  static constexpr int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  static constexpr int SHARED_MEM = TILE_SIZE * sizeof(float);
};

__global__ void bitonicMergeStep(float* data, int n, int j, int k) {
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

  sortTilesUnrolledKernel<TILE_SIZE>
      <<<NUM_TILES, BLOCK_SIZE, SHARED_MEM>>>(d_data, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int k = 2; k <= N; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      bitonicMergeStep<<<GRID_SIZE, BLOCK_SIZE>>>(d_data, N, j, k);
      CUDA_CHECK(cudaDeviceSynchronize());
    }
  }
}

extern "C" void bitonicSortSharedMemory(float* h_data, int n) {
  float* d_data;
  size_t bytes = n * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_data, bytes));
  CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

  switch (n) {
    case 128:
      sort_fixed_size<128>(d_data);
      break;
    case 512:
      sort_fixed_size<512>(d_data);
      break;
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
      fprintf(stderr, "Unsupported size %d\n", n);
      cudaFree(d_data);
      std::exit(EXIT_FAILURE);
  }

  CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_data));
}

extern "C" void bitonicSortCPU(float* data, int n) {
  std::sort(data, data + n);
}