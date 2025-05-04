#include <mpi.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

double simple_det(vector<double> A, int n) {
    double det = 1.0;
    int sign = 1;

    for (int k = 0; k < n; ++k) {
        int pivot = k;
        double max_val = fabs(A[k * n + k]);

        for (int i = k + 1; i < n; ++i) {
            double val = fabs(A[i * n + k]);
            if (val > max_val) {
                max_val = val;
                pivot = i;
            }
        }

        if (fabs(max_val) < 1e-12) return 0.0;

        if (pivot != k) {
            for (int j = 0; j < n; ++j)
                swap(A[k * n + j], A[pivot * n + j]);
            sign *= -1;
        }

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; ++j)
                A[i * n + j] -= factor * A[k * n + j];
        }
    }

    for (int i = 0; i < n; ++i)
        det *= A[i * n + i];

    return sign * det;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int n;
    vector<double> A_flat;

    if (rank == 0) {
        n = 840;
        cout << "Розмір матриці: " << n << "." << endl;

        A_flat.resize(n * n);
        mt19937 gen(11);
        uniform_real_distribution<> dis(0.0, 1.0);
        for (int i = 0; i < n * n; i++) {
            A_flat[i] = dis(gen);
        }
    }

    vector<double> A_original;
    if (rank == 0) {
        A_original = A_flat;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n % numProcs != 0) {
        if (rank == 0)
            cout << "Розмір матриці n має націло ділитись на к-сть процесів." << endl;
        MPI_Finalize();
        return -1;
    }
    int n_local = n / numProcs;
    vector<double> local_A(n_local * n, 0.0);

    MPI_Scatter(A_flat.data(), n_local * n, MPI_DOUBLE,
                local_A.data(), n_local * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double det = 1.0;
    int local_swaps = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        double local_max = 0.0;
        int local_candidate_index = -1;
        for (int i = 0; i < n_local; i++) {
            int global_i = rank * n_local + i;
            if (global_i >= k) {
                double val = fabs(local_A[i * n + k]);
                if (val > local_max) {
                    local_max = val;
                    local_candidate_index = global_i;
                }
            }
        }

        struct {
            double value;
            int index;
        } local_candidate{}, global_candidate{};

        local_candidate.value = local_max;
        local_candidate.index = (local_candidate_index == -1 ? -1 : local_candidate_index);

        MPI_Allreduce(&local_candidate, &global_candidate, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        int pivot_owner = global_candidate.index / n_local;
        int pivot_local_i = global_candidate.index % n_local;

        int current_owner = k / n_local;
        int local_k_i = k % n_local;

        if (global_candidate.index != k) {
            if (current_owner == pivot_owner) {
                if (rank == current_owner) {
                    for (int j = 0; j < n; j++) {
                        swap(local_A[local_k_i * n + j], local_A[pivot_local_i * n + j]);
                    }
                    local_swaps++;
                }
            }
            else {
                if (rank == current_owner) {
                    vector<double> temp_row(n);
                    MPI_Sendrecv(&local_A[local_k_i * n], n, MPI_DOUBLE,
                                 pivot_owner, 0,
                                 temp_row.data(), n, MPI_DOUBLE,
                                 pivot_owner, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int j = 0; j < n; j++) {
                        local_A[local_k_i * n + j] = temp_row[j];
                    }
                    local_swaps++;
                }
                else if (rank == pivot_owner) {
                    vector<double> temp_row(n);
                    MPI_Sendrecv(&local_A[pivot_local_i * n], n, MPI_DOUBLE,
                                 current_owner, 0,
                                 temp_row.data(), n, MPI_DOUBLE,
                                 current_owner, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    for (int j = 0; j < n; j++) {
                        local_A[pivot_local_i * n + j] = temp_row[j];
                    }
                }
            }
        }

        vector<double> pivot_row(n, 0.0);
        if (rank == current_owner) {
            for (int j = 0; j < n; j++) {
                pivot_row[j] = local_A[local_k_i * n + j];
            }
        }
        MPI_Bcast(pivot_row.data(), n, MPI_DOUBLE, current_owner, MPI_COMM_WORLD);

        if (fabs(pivot_row[k]) < 1e-12) {
            if (rank == 0) {
                cout << "Матриця сингулярна. Визначник = 0.0." << endl;
            }
            MPI_Finalize();
            return 0;
        }

        for (int i = 0; i < n_local; i++) {
            int global_i = rank * n_local + i;
            if (global_i > k) {
                double factor = local_A[i * n + k] / pivot_row[k];
                for (int j = k; j < n; j++) {
                    local_A[i * n + j] -= factor * pivot_row[j];
                }
            }
        }
    }

    vector<double> A_final;
    if (rank == 0) {
        A_final.resize(n * n, 0.0);
    }
    MPI_Gather(local_A.data(), n_local * n, MPI_DOUBLE,
               A_final.data(), n_local * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    int total_swaps = 0;
    MPI_Reduce(&local_swaps, &total_swaps, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            det *= A_final[i * n + i];
        }
        int sign = (total_swaps % 2 == 0) ? 1 : -1;
        det = sign * det;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        cout << "Визначник матриці з MPI-алгоритму: " << setprecision(10) << det << '.' << endl;
        cout << "Час виконання MPI-алгоритму: " << elapsed.count() << " секунд." << endl;
        cout << '\n';
    }

    if (rank == 0) {
        auto start_simple_det = std::chrono::high_resolution_clock::now();

        double det_ref = simple_det(A_original, n);

        auto end_simple_det = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_simple_det = end_simple_det - start_simple_det;

        cout << "Визначник матриці з простого алгоритму: " << setprecision(10) << det_ref << endl;
        double diff = fabs(det - det_ref);
        cout << "Різниця між визначниками алгоритмів = " << diff << "." << endl;
        if (diff < 1e-6)
            cout << "Визначник ОБЧИСЛЕНО ПРАВИЛЬНО (у межах допустимої похибки)." << endl;
        else
            cout << "Визначник неправильний (занадто велика різниця) АБО різницю не порахувати (nan)" << endl;
        cout << "Час виконання простого алгоритму: " << elapsed_simple_det.count() << " секунд." << endl;
    }

    MPI_Finalize();
    return 0;
}