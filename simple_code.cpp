#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

// Обчислення визначника методом Гаусса з частковим вибором головного елемента.
double determinant(vector<double> A, int n) {
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

int main() {
    int n = 2520;
    cout << "Розмірність квадратної матриці: " << n << endl;

    vector<double> A (n*n);

    mt19937 gen(11);
    uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < n * n; i++) {
        A[i] = dis(gen);
    }

    vector<double> A_copy = A;

    auto start= std::chrono::high_resolution_clock::now();
    double det = determinant(A_copy, n);
    auto end= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    cout << "Визначник матриці: " << setprecision(10) << det << "." << endl;
    cout << "Час виконання алгоритму: " << diff.count() << " секунд." << endl;

    return 0;
}