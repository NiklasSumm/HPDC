#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 33554432  // Arraygröße (muss Potenz von 2 sein)
#define TILE_S 1024

__global__ void preSort_shared(float* data){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_data[TILE_S];

    shared_data[threadIdx.x] = data[tid];

    __syncthreads();

    for (int k = 2; k <= TILE_S; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = threadIdx.x ^ j;
            if (partner > threadIdx.x && partner < TILE_S) {
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

__global__ void sort_shared(float* data, int j_start, int k){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float shared_data[TILE_S];

    shared_data[threadIdx.x] = data[tid];

    __syncthreads();

    for (int j = j_start >> 1; j > 0; j >>= 1) {
        int partner = threadIdx.x ^ j;
        if (partner > threadIdx.x && partner < TILE_S) {
            bool asc = ((threadIdx.x & k) == 0);
            if ((shared_data[threadIdx.x] > shared_data[partner]) == asc) {
                float tmp = shared_data[threadIdx.x];
                shared_data[threadIdx.x] = shared_data[partner];
                shared_data[partner] = tmp;
            }
        }
        if (partner >= TILE_S) printf("error");
        __syncthreads();
    }

    data[tid] = shared_data[threadIdx.x];
}

__global__ void bitonicSortIterative(float* data, int n, int j, int k) {
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

template <int TILE_SIZE, int K = 2>
__global__ void sortTilesUnrolledKernel(float* data, int n) {
  __shared__ float shared_data[TILE_SIZE];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int tile_start = bid * TILE_SIZE;

  if (tile_start + tid < n) {
    shared_data[tid] = data[tile_start + tid];
  } else {
    shared_data[tid] = 9999999;
  }
  __syncthreads();

  bitonicSortUnrolled<TILE_SIZE>(shared_data, tid);

  if (tile_start + tid < n) {
    data[tile_start + tid] = shared_data[tid];
  }
}

struct BitonicSortConfig {
  static_assert((N & (N - 1)) == 0, "N must be power of two");
  static constexpr int TILE_SIZE = 512;
  static constexpr int BLOCK_SIZE = TILE_SIZE;
  static constexpr int NUM_TILES = N / TILE_SIZE;
  static constexpr int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
  static constexpr int SHARED_MEM = TILE_SIZE * sizeof(float);
};

void checkCuda(cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main() {
    float* h_data = (float*)malloc(N * sizeof(float));
    float* d_data;

    // Zufällige Werte initialisieren
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 1000);
    }

    // Speicher auf der GPU reservieren
    checkCuda(cudaMalloc((void**)&d_data, N * sizeof(float)), "cudaMalloc");

    // Daten auf die GPU kopieren
    checkCuda(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");

    // CUDA Events für Zeitmessung erstellen
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    // Startzeit erfassen
    checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start");

//    using Config = BitonicSortConfig;
//
//    constexpr int TILE_SIZE = Config::TILE_SIZE;
//    constexpr int BLOCK_SIZE = Config::BLOCK_SIZE;
//    constexpr int NUM_TILES = Config::NUM_TILES;
//    constexpr int GRID_SIZE = Config::GRID_SIZE;
//    constexpr int SHARED_MEM = Config::SHARED_MEM;
//
//    sortTilesUnrolledKernel<TILE_SIZE>
//      <<<NUM_TILES, BLOCK_SIZE, SHARED_MEM>>>(d_data, N);
//    checkCuda(cudaDeviceSynchronize(), "Kernel1 execution");

    preSort_shared<<<N / TILE_S, TILE_S>>>(d_data);
    checkCuda(cudaDeviceSynchronize(), "Pre-Sort Kernel execution");

    // Bitonic Sort Kernel-Aufrufe
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int k = TILE_S; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            //if (j <= (TILE_S >> 1)){
            //    sort_shared<<<N / TILE_S, TILE_S>>>(d_data, j, k);
            //    checkCuda(cudaDeviceSynchronize(), "Pre-Sort Kernel execution");
            //    break;
            //}

            bitonicSortIterative<<<numBlocks, threadsPerBlock>>>(d_data, N, j, k);
            checkCuda(cudaDeviceSynchronize(), "Pre-Sort Kernel execution");
        }
    }

    // Stoppzeit erfassen
    checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    // Zeit berechnen
    float elapsedTime;
    checkCuda(cudaEventElapsedTime(&elapsedTime, start, stop), "cudaEventElapsedTime");

    printf("Bitonic Sort auf der GPU für %d Elemente dauerte: %.3f ms\n", N, elapsedTime);

    // Daten zurück auf Host kopieren (optional)
    checkCuda(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy DeviceToHost");


    // Korrektheitsprüfung
    bool isSorted = true;
    for (int i = 0; i < N - 1; i++) {
        if (h_data[i] > h_data[i + 1]) {
            printf("Sortierfehler an Position %d: %.3f > %.3f\n", i, h_data[i], h_data[i + 1]);
            isSorted = false;
            break;
        }
    }

    if (isSorted) {
        printf("Sortierergebnis korrekt.\n");
    } else {
        printf("Sortierergebnis fehlerhaft.\n");
    }

    // Sortiertes Array ausgeben
//    printf("Sortiertes Array:\n");
//    for (int i = 0; i < N; i++) {
//        printf("%.3f ", h_data[i]);
//        if ((i + 1) % 16 == 0) printf("\n");  // Zeilenumbruch nach 16 Werten
//    }
//    printf("\n");

    // Events und Speicher freigeben
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
