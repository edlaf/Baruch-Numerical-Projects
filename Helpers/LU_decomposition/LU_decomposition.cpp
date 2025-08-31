#include "LU_decomposition.hpp"

LU::LU(Matrix A):A_(A), L(Matrix::zeros(A.rows(), A.rows())), U(Matrix::zeros(A.rows(), A.rows())), P(Matrix::eye(A.rows())), n_(A.rows()){}

// Doolittle decomp
std::vector<Matrix> LU::decompose_without_pivot(){
    L = Matrix::zeros(n_, n_);
    U = Matrix::zeros(n_, n_);
    for (size_t i = 0; i < n_; i++){
        L(i,i) = 1.0;
        for (size_t j = i; j < n_; j++){
            double sum = 0.0;
            for (size_t k = 0; k < i; k++) {
                sum += L(i, k) * U(k, j);
            }
            U(i,j) = A_(i,j) - sum;
        }

        for (size_t j = i+1; j < n_; j++){
            double sum = 0.0;
            for (size_t k = 0; k < i; k++) {
                sum += L(j, k) * U(k, i);
            }
            L(j,i) = (A_(j, i) - sum)/U(i,i);
        }
    }
    return {L, U};
}

std::vector<Matrix> LU::decompose_with_pivot() {
    Matrix A = A_;
    L = Matrix::zeros(n_, n_);
    U = Matrix::zeros(n_, n_);
    P = Matrix::eye(n_);
    for (size_t i = 0; i < n_; i++) {
        double max_val = std::abs(A(i,i));
        size_t pivot = i;
        for (size_t k = i+1; k < n_; k++) {
            if (std::abs(A(k,i)) > max_val) {
                max_val = std::abs(A(k,i));
                pivot = k;
            }
        }

        if (pivot != i) {
            A.swapRows(i, pivot);
            P.swapRows(i, pivot);
            for (size_t k = 0; k < i; k++) {
                L.swapElements(i, k, pivot, k);
            }
        }

        for (size_t j = i; j < n_; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < i; k++) {
                sum += L(i,k) * U(k,j);
            }
            U(i,j) = A(i,j) - sum;
        }

        L(i,i) = 1.0;
        for (size_t j = i+1; j < n_; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < i; k++) {
                sum += L(j,k) * U(k,i);
            }
            L(j,i) = (A(j,i) - sum) / U(i,i);
        }
    }

    return {P, L, U};
}

double LU::verify_(){
    return (A_ - P*L*U).norm();
}