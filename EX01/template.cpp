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

  MPI_Status status;
  int dummy = 1;

  // Get the rank and size of the MPI communicator
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double start_time = MPI_Wtime();
  for (int iteration = 0; iteration < 10; iteration++){
    MPI_Barrier(MPI_COMM_WORLD);
  }
  double end_time = MPI_Wtime();

  if (rank == 0) {
    double total_time = (end_time - start_time) * 1000;
    printf("%i Processes - Total time: %f ms\n", size, total_time);
  }

  // Finalize MPI
  MPI_Finalize();
  return 0;
}
