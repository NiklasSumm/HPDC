/* HPDC SS 2025, C++ Template 
 */
#include <mpi.h>
#include <random>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char *argv[]) {

  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Get the rank and size of the MPI communicator
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 1){
    double matrixA[2048][2048];
    double matrixB[2048][2048];
    double matrixC[2048][2048];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-10, 10);

    for (int i = 0; i < 2048; i++){
      for (int j = 0; j < 2048; j++){
        matrixA[i][j] = dis(gen);
        matrixB[i][j] = dis(gen);
      }
    }

    double start_time = MPI_Wtime();

    for (int iterations = 0; iterations < 10; iterations++){
      for (int i = 0; i < 2048; i++){
        for (int j = 0; j < 2048; j++){
          double sum = 0;
          for (int k = 0; k < 2048; k++){
            sum += matrixA[k][j] * matrixB[i][k];
          }
          matrixC[i][j] = sum;
        }
      }
    }

    double end_time = MPI_Wtime();

    printf("Matrix multiply needed %f ms", 1000*(end_time - start_time)/10);
  }

  // Finalize MPI
  MPI_Finalize();
  return 0;
}
