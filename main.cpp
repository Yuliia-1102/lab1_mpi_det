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
        n = 2520;
        cout << "Розмір матриці: " << n << "." << endl;

        A_flat.resize(n * n);
        mt19937 gen(11);
        uniform_int_distribution<> dis(1, 9);
        for (int i = 0; i < n * n; ++i) {
            A_flat[i] = dis(gen) / 10.0;
        }

        /*for (int i = 0; i < n; ++i) { // виведення матриці
            for (int j = 0; j < n; ++j) {
                cout << A_flat[i * n + j] << " ";
            }
            cout << endl;
        }*/
    }

    vector<double> A_original;
    if (rank == 0) {
        A_original = A_flat;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* if (n % numProcs != 0) {
        if (rank == 0)
            cout << "Розмір матриці n має націло ділитись на к-сть процесів." << endl;
        MPI_Finalize();
        return -1;
    }
    int n_local = n / numProcs; // к-сть рядків що отримує кожен процес
    vector<double> local_A(n_local * n, 0.0);

    MPI_Scatter(A_flat.data(), n_local * n, MPI_DOUBLE,
                local_A.data(), n_local * n, MPI_DOUBLE,
                0, MPI_COMM_WORLD); */

    vector<int> elements_count(numProcs);
    vector<int> displs(numProcs);

    if (rank == 0) {
        int base_rows = n / numProcs; // к-сть рядків, що отримає точно кожен процес
        int remainder = n % numProcs; // к-сть рядків, що треба розділити по процесах - перші процеси їх отримують поступово по одному

        int offset = 0;
        for (int i = 0; i < numProcs; ++i) {
            int rows = base_rows + (i < remainder ? 1 : 0); // к-сть рядків, що піде процесу
            elements_count[i] = rows * n; // к-сть елементів, що піде процесу
            displs[i] = offset;
            offset += elements_count[i]; // оновлюємо зсув глобальної матриці поелементно
        }
    }
    MPI_Bcast(elements_count.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displs.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

    int n_local = elements_count[rank] / n; // к-сть рядків, що іде кожному процесу
    vector<double> local_A(n_local * n, 0.0);

    // для кожного процесу: з глобальної матриці з 0 процесу беремо поступово: elements_count - 10 елементів (к-сть),
    // displs - від 0 індексу та записуємо в локальну матрицю з к-стю елементів - 10.
    MPI_Scatterv(A_flat.data(), elements_count.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), elements_count[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    vector<int> row_counts(numProcs), row_displs(numProcs);
    if (rank == 0) {
        for (int p = 0; p < numProcs; ++p)
            row_counts[p] = elements_count[p] / n; // масив - зберігає к-сть рядків у кожному процесі
        row_displs[0] = 0;
        for (int p = 1; p < numProcs; ++p)
            row_displs[p] = row_displs[p - 1] + row_counts[p - 1]; // масив - зберігає зсув (у рядках) глобальної матриці; з якого рядка починаються рядки процесу
    }
    MPI_Bcast(row_counts.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(row_displs.data(), numProcs, MPI_INT, 0, MPI_COMM_WORLD);

    double det = 1.0;
    int local_swaps = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        // Кожен процес у своїй локальній матриці шукає локального кандидата для головного елемента у кожному стовпці
        double local_max = 0.0;
        int local_candidate_index = -1;
        for (int i = 0; i < n_local; i++) {
            int global_i = row_displs[rank] + i;;
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

        auto owner_of = [&](int g){
            int j = upper_bound(row_displs.begin(), row_displs.end(), g) - row_displs.begin() - 1;
            return j;
        };

        int pivot_owner = owner_of(global_candidate.index); // процес, якому належить рядок з головним елементом
        int pivot_local_i = global_candidate.index - row_displs[pivot_owner];

        int current_owner = owner_of(k); // процес, якому належить поточний крок (рядок k)
        int local_k_i = k - row_displs[current_owner];

        if (global_candidate.index != k) { // обмін рядками
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

        for (int i = 0; i < n_local; i++) { // кожен процес оновлює свої локальні рядки - зануляє решту певного стовпця під головним елементом
            int global_i = row_displs[rank] + i;
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
    MPI_Gatherv(local_A.data(), elements_count[rank], MPI_DOUBLE,
                A_final.data(), elements_count.data(), displs.data(), MPI_DOUBLE,
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