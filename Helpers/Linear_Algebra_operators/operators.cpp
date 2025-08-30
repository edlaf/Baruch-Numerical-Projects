#include "operators.hpp"

Matrix LU_decompose(const Matrix& A) {
    LU lu(A);
    return lu.decompose();
}

Matrix Cholesky_decompose(const Matrix& A) {
    Cholesky chol(A);
    return chol.decompose();
}

Matrix inverse(const Matrix& A) {
    Inverse inv(A);
    return inv.compute();
}
