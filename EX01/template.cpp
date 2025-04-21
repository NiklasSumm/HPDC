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

  for (int size = 1024; size <= 1024*1024; size *= 2){
    double start_time = MPI_Wtime();

    for (int i = 0; i < 10; i++){
      if (rank == 0) {
        MPI_Send(data, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(data, size, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      } else if (rank == 1) {
        MPI_Recv(data, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(data, size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
      }
    }

    double end_time = MPI_Wtime();

    if (rank == 0) {
      double total_time = end_time - start_time;
      double avg_round_trip = (total_time / 10) * 1000; // in ms
      double half_trip = avg_round_trip / 2;
      printf("Message size: %6d bytes | Full RTT: %.6f ms | Half RTT (Latency): %.6f ms\n",
             size, avg_round_trip, half_trip);
    }
  }

  // Finalize MPI
  MPI_Finalize();
  return 0;
}
