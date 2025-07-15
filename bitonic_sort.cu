#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024  // Arraygröße (muss Potenz von 2 sein)

__global__ void bitonicSortKernel(float* data, int n, int j, int k) {
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

    // Bitonic Sort Kernel-Aufrufe
    int threadsPerBlock = 256;
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    //int totalStages = 0;
    //for (int temp = N; temp > 1; temp >>= 1) totalStages++;

    //for (int stage = 1; stage <= totalStages; stage++) {
    //    for (int passOfStage = 1; passOfStage <= stage; passOfStage++) {
    //        bitonicSortKernel<<<numBlocks, threadsPerBlock>>>(d_data, stage, passOfStage);
    //        checkCuda(cudaGetLastError(), "Kernel execution");
    //    }
    //}

    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicSortKernel<<<numBlocks, threadsPerBlock>>>(d_data, N, j, k);
            checkCuda(cudaGetLastError(), "Kernel execution");
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

    // Events und Speicher freigeben
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);

    return 0;
}