/* HPDC SS 2025, C++ Template 
 */
#include <mpi.h>

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

  char *data = (char *)malloc(1024*1024);
  if (!data) {
    fprintf(stderr, "Failed to allocate buffer\n");
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  bool blocking = true;
  MPI_Request request;

  for (int size = 1024; size <= 1024*1024; size *= 2){
    double start_time = MPI_Wtime();

    if (blocking){
      for (int i = 0; i < 10; i++){
        if (rank == 0) {
          MPI_Send(data, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        } else if (rank == 1) {
          MPI_Recv(data, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }
    else{
      for (int i = 0; i < 10; i++){
        if (rank == 0) {
          MPI_Isend(data, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &request);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
          MPI_Irecv(data, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &request);
          MPI_Wait(&request, MPI_STATUS_IGNORE);
        } 
      }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
      double time = end_time - start_time;
      double total_bytes = (double)size * 10;
      double bandwidth = (total_bytes / time) / (1024 * 1024);
      printf("Non-Blocking | Message size: %6d bytes | Bandwidth: %.2f MB/s\n", size, bandwidth);
    }
  }

  // Finalize MPI
  MPI_Finalize();
  return 0;
}
