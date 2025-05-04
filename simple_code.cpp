#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
using namespace std;

// Обчислення визначника методом Гаусса з частковим вибором головного елемента.
double determinant(vector<vector<double>> A) {
    size_t n = A.size();
    double det = 1.0;
    int sign = 1;

    for (int k = 0; k < n; ++k) {
        int row = k;
        double max_val = fabs(A[k][k]);
        for (int i = k + 1; i < n; ++i) {
            if (fabs(A[i][k]) > max_val) {
                max_val = fabs(A[i][k]);
                row = i;
            }
        }

        if (fabs(A[row][k]) < 1e-12) {
            return 0.0;
        }

        if (row != k) {
            swap(A[k], A[row]);
            sign = -sign;
        }

        det *= A[k][k];

        for (int i = k + 1; i < n; ++i) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < n; ++j) {
                A[i][j] = A[i][j] - factor * A[k][j];
            }
        }
    }
    return sign * det;
}

int main() {
    int n;
    cout << "Введіть розмірність квадратної матриці: ";
    cin >> n;

    vector<vector<double>> A(n, vector<double>(n));

    mt19937 gen(11);
    uniform_real_distribution<> dis(0.0, 1.0);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            A[i][j] = dis(gen);
        }
    }

    auto start= std::chrono::high_resolution_clock::now();
    double det = determinant(A);
    auto end= std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    cout << "Визначник матриці: " << setprecision(10) << det << "." << endl;
    cout << "Час виконання алгоритму: " << diff.count() << " секунд." << endl;

    return 0;
}