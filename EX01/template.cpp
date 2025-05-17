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

    int N = 5;

    // Argumente parsen
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "matrix_size=", 12) == 0) {
            N = atoi(argv[i] + 12);
        }
    }

    if (N == 0) {
        if (rank == 0)
            cerr << "Usage: " << argv[0] << " matrix_size=<N>" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    vector<vector<int>> A, B;
    if (rank == 0) {
        A.resize(N, vector<int>(N));
        B.resize(N, vector<int>(N));
        initializeMatrices(A, B, N);
    }

    int* flatB = new int[N * N];

    // Broadcast Matrix B an alle
    if (rank == 0) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                flatB[i * N + j] = B[i][j];
    }
    MPI_Bcast(flatB, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    // Scatterv Vorbereitung
    vector<int> sendcounts(size);
    vector<int> displs(size);
    int rowsPerProc = N / size;
    int remainder = N % size;

    int offset = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (rowsPerProc + (i < remainder ? 1 : 0)) * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    int localRows = sendcounts[rank] / N;

    // Speicher für lokalen Teil
    int* localA = new int[sendcounts[rank]];
    int* localC = new int[sendcounts[rank]];

    // Flat-A nur auf Rank 0
    int* flatA = nullptr;
    int* flatC = nullptr;
    if (rank == 0) {
        flatA = new int[N * N];
        flatC = new int[N * N];
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                flatA[i * N + j] = A[i][j];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double startTime = MPI_Wtime();

    // A verteilen
    MPI_Scatterv(flatA, sendcounts.data(), displs.data(), MPI_INT,
                 localA, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Teilweise Matrixmultiplikation
    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < N; ++j) {
            localC[i * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                localC[i * N + j] += localA[i * N + k] * flatB[k * N + j];
            }
        }
    }

    // Ergebnisse sammeln
    MPI_Gatherv(localC, sendcounts[rank], MPI_INT,
                flatC, sendcounts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double endTime = MPI_Wtime();

    if (rank == 0) {
        cout << "Matrix C = A * B (Größe: " << N << "x" << N << ")" << endl;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j)
                cout << flatC[i * N + j] << "\t";
            cout << endl;
        }
        cout << "Gesamtdauer inkl. Scatterv/Gatherv/Broadcast/Multiplikation: "
             << (endTime - startTime) << " Sekunden." << endl;
    }

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
