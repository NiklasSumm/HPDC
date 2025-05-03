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
    if (rank == 0) {
      // Root waits for notification from all other processes
      for (int i = 1; i < size; i++) {
          MPI_Recv(&dummy, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      }
      // Root sends release signal to all processes
      for (int i = 1; i < size; i++) {
          MPI_Send(&dummy, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
      }
    } else {
      // Notify root
      MPI_Send(&dummy, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      // Wait for release signal
      MPI_Recv(&dummy, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
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
