#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int rank, size;
    int array_size = 12; // Muss durch s teilbar sein
    int s = 2;           // Anzahl der Teile (Chunks)

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (array_size % s != 0) {
        if (rank == 0) {
            printf("array_size muss durch s teilbar sein!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int chunk_size = array_size / s;

    // Array anlegen und mit array[i] = i initialisieren
    float* array = (float*)malloc(array_size * sizeof(float));
    for (int i = 0; i < array_size; i++) {
        array[i] = (float)i;
    }

    // Speicher für Empfangschunk
    float* recv_chunk = (float*)malloc(chunk_size * sizeof(float));

    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    // Iteration über size Runden
    for (int iter = 0; iter < size; iter++) {
        // Bestimme Chunk-Index für diese Iteration beim eigenen Prozess
        int send_chunk_index = (rank - iter + s) % s;
        float* send_ptr = array + send_chunk_index * chunk_size;

        // Zuerst senden
        MPI_Send(send_ptr, chunk_size, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);

        // Dann empfangen
        MPI_Recv(recv_chunk, chunk_size, MPI_FLOAT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int recv_chunk_index = (rank - 1 + iter + s) % s;
        for (int i = 0; i < chunk_size; i++) {
            array[recv_chunk_index * chunk_size + i] += recv_chunk[i];
        }
    }

    // Ergebnis ausgeben
    printf("Prozess %d hat folgendes Array nach %d Iterationen:\n", rank, size);
    for (int i = 0; i < array_size; i++) {
        printf("%.1f ", array[i]);
    }
    printf("\n");

    // Speicher freigeben
    free(array);
    free(recv_chunk);

    MPI_Finalize();
    return 0;
}
