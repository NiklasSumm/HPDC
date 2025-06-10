#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8   // Matrixgröße N x N
#define iterations 100
#define MASTER 0

void print_matrix(double* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%5.1f ", matrix[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    int rank, size = 1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == MASTER) {
        if (argc > 1) {
            //iterations = atoi(argv[1]);
            //if (iterations <= 0) iterations = 1;
        }

        printf("Stencil-Berechnung mit %d Iteration(en)\n\n", iterations);

        double* matrix = (double*)malloc(N * N * sizeof(double));
        double* result_matrix = (double*)malloc(N * N * sizeof(double));

        // Initialisierung: oberste Zeile erstes Viertel 127, Rest 0
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == 0 && j < N/4)
                    matrix[i*N + j] = 127.0;
                else
                    matrix[i*N + j] = 0.0;
            }
        }

        printf("Initiale Matrix:\n");
        print_matrix(matrix, N);

        // Zeitmessung starten
        double start_time = MPI_Wtime();

        // Iterationen
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    double center = matrix[i*N + j];
                    double top    = (i == 0)   ? 0.0 : matrix[(i-1)*N + j];
                    double bottom = (i == N-1) ? 0.0 : matrix[(i+1)*N + j];
                    double left   = (j == 0)   ? 0.0 : matrix[i*N + (j-1)];
                    double right  = (j == N-1) ? 0.0 : matrix[i*N + (j+1)];

                    result_matrix[i*N + j] = center + 0.24 * (-4 * center + top + bottom + left + right);
                }
            }

            // Ergebnis der Iteration übernehmen
            double* tmp = matrix;
            matrix = result_matrix;
            result_matrix = tmp;
        }

        // Zeitmessung stoppen
        double end_time = MPI_Wtime();

        // Ergebnis ausgeben
        printf("Ergebnis nach %d Iterationen:\n", iterations);
        if (iterations % 2 == 0)
            print_matrix(matrix, N);
        else
            print_matrix(result_matrix, N);

        printf("Berechnungszeit: %f Sekunden\n", end_time - start_time);

        free(matrix);
        free(result_matrix);
    }

    MPI_Finalize();
    return 0;
}
