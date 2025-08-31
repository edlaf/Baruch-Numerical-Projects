#include "operators.hpp"

std::vector<Matrix> LU_decompose_with_pivot(const Matrix& A) {
    LU lu(A);
    return lu.decompose_with_pivot();
}

std::vector<Matrix> LU_decompose_without_pivot(const Matrix& A) {
    LU lu(A);
    return lu.decompose_without_pivot();
}

Matrix Cholesky_decompose(const Matrix& A) {
    Cholesky chol(A);
    return chol.decompose();
}

Matrix inverse(const Matrix& A) {
    Inverse inv(A);
    return inv.compute();
}
