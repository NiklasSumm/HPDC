#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>

using namespace std;

void initializeMatrices(vector<vector<int>>& A, vector<vector<int>>& B, int N) {
    for (int i = 0; i < N; i++) {
        vector<int> rowA(N), rowB(N);
        for (int j = 0; j < N; j++) {
            rowA[j] = i + j;
            rowB[j] = i * j;
        }
        A[i] = rowA;
        B[i] = rowB;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 512;

    // Argumente parsen
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "matrix_size=", 12) == 0) {
            N = atoi(argv[i] + 12);
        }
    }

    if (N % size != 0) {
        if (rank == 0)
            cerr << "Matrixgröße N muss durch Anzahl der Prozesse teilbar sein!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<vector<int>> A, B;
    if (rank == 0) {
        A.resize(N, vector<int>(N));
        B.resize(N, vector<int>(N));
        initializeMatrices(A, B, N);
    }

    int* flatB = new int[N * N];
    int* localA = new int[(N / size) * N];
    int* localC = new int[(N / size) * N];

    // Startzeit
    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // B an alle senden
    if (rank == 0) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                flatB[i * N + j] = B[i][j];
    }
    MPI_Bcast(flatB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // A verteilen
    int* flatA = nullptr;
    if (rank == 0) {
        flatA = new int[N * N];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                flatA[i * N + j] = A[i][j];
    }
    MPI_Scatter(flatA, (N / size) * N, MPI_INT, localA, (N / size) * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Teilweise Matrixmultiplikation
    for (int i = 0; i < N / size; i++) {
        for (int j = 0; j < N; j++) {
            localC[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                localC[i * N + j] += localA[i * N + k] * flatB[k * N + j];
            }
        }
    }

    // Ergebnisse sammeln
    int* flatC = nullptr;
    if (rank == 0)
        flatC = new int[N * N];
    MPI_Gather(localC, (N / size) * N, MPI_INT, flatC, (N / size) * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Endzeit
    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    // Ausgabe
    if (rank == 0) {
        cout << "Matrix C = A * B (Größe: " << N << "x" << N << ")" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                cout << flatC[i * N + j] << "\t";
            cout << endl;
        }
        cout << "Gesamtdauer inkl. Scatter/Gather/Broadcast/Multiplikation: "
             << (endTime - startTime) << " Sekunden." << endl;
    }

    // Speicher freigeben
    delete[] flatB;
    delete[] localA;
    delete[] localC;
    if (rank == 0) {
        delete[] flatA;
        delete[] flatC;
    }

    MPI_Finalize();
    return 0;
}
