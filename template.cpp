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

  // Get the hostname of the processor
  char hostname[MPI_MAX_PROCESSOR_NAME];
  int hostname_len;
  MPI_Get_processor_name(hostname, &hostname_len);

  // Output the rank, size, hostname, and all CLI parameters
  std::cout << "Hello from rank " << rank << " out of " << size << " processes on host " << hostname
            << " with cli parameters: '";
  for (int i = 1; i < argc; ++i) {
    std::cout << argv[i];
    if (i < argc - 1) std::cout << ", ";
  }
  std::cout << "'" << std::endl;

  // Finalize MPI
  MPI_Finalize();
  return 0;
}
