#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

void ring_allreduce(float* array, int array_size, int s, int rank, int size) {
    int chunk_size = array_size / s;
    float* recv_chunk = (float*)malloc(chunk_size * sizeof(float));
    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    // Reduce-Scatter Phase
    for (int iter = 0; iter < s; iter++) {
        int send_chunk_index = (rank - iter + s) % s;
        float* send_ptr = array + send_chunk_index * chunk_size;

        MPI_Send(send_ptr, chunk_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Recv(recv_chunk, chunk_size, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int recv_chunk_index = (rank - (iter + 1) + s) % s;
        for (int i = 0; i < chunk_size; i++) {
            array[recv_chunk_index * chunk_size + i] += recv_chunk[i];
        }
    }

    // Allgather Phase
    for (int iter = 0; iter < s; iter++) {
        int send_chunk_index = (rank - iter + 1 + s) % s;
        float* send_ptr = array + send_chunk_index * chunk_size;

        MPI_Send(send_ptr, chunk_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
        MPI_Recv(recv_chunk, chunk_size, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int recv_chunk_index = (rank - iter + s) % s;
        for (int i = 0; i < chunk_size; i++) {
            array[recv_chunk_index * chunk_size + i] = recv_chunk[i];
        }
    }

    free(recv_chunk);
}

void native_allreduce(float* array, int array_size) {
    float* result = (float*)malloc(array_size * sizeof(float));
    MPI_Allreduce(array, result, array_size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    for (int i = 0; i < array_size; i++) {
        array[i] = result[i];
    }
    free(result);
}

int main(int argc, char** argv) {
    int rank, size;
    int array_size, s;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Usage: %s <array_size> <s>\n", argv[0]);
        }
        //MPI_Abort(MPI_COMM_WORLD, 1);
    }

    array_size = atoi(argv[1]);
    s = atoi(argv[2]);

    if (array_size % s != 0 || size != s) {
        if (rank == 0) {
            printf("Fehler: array_size muss durch s teilbar sein und Anzahl Prozesse == s!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int runs = 5;
    double total_time_ring = 0.0, total_time_native = 0.0;

    for (int run = 0; run < runs; run++) {
        float* array = (float*)malloc(array_size * sizeof(float));
        for (int i = 0; i < array_size; i++) {
            array[i] = (float)i;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        ring_allreduce(array, array_size, s, rank, size);
        double end = MPI_Wtime();
        total_time_ring += (end - start);

        free(array);
    }

    for (int run = 0; run < runs; run++) {
        float* array = (float*)malloc(array_size * sizeof(float));
        for (int i = 0; i < array_size; i++) {
            array[i] = (float)i;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        native_allreduce(array, array_size);
        double end = MPI_Wtime();
        total_time_native += (end - start);

        free(array);
    }

    if (rank == 0) {
        printf("\nParameter: array_size = %d, s = %d, Prozesse = %d\n", array_size, s, size);
        printf("Durchschnittliche Zeiten über %d Läufe:\n", runs);
        printf("Ring-Allreduce:   %.6f Sekunden\n", total_time_ring / runs);
        printf("MPI_Allreduce:    %.6f Sekunden\n\n", total_time_native / runs);
    }

    MPI_Finalize();
    return 0;
}
