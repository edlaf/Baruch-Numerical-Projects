#include "Cholesky.hpp"

Cholesky::Cholesky(Matrix A):A_(A) ,L(Matrix::zeros(A.rows(), A.rows())), n_(A.rows()){}

Matrix Cholesky::decompose(){
    for (size_t i = 0; i < n_; i++){
        double sum = 0.0;
        for (size_t k = 0; k < i; k++) {
            sum += L(i, k) * L(i, k);
        }
        L(i,i) = std::sqrt(A_(i,i) - sum);
        for (size_t j = i+1; j<n_; j++){
            double prod = 0.0;
            for (size_t k = 0; k < i; k++) {
                prod += L(j, k) * L(i, k);
            }
            L(j,i) = (A_(j,i) - prod) / L(i,i);
        }
    }
    return L;
}

double Cholesky::verify_Cholesky(){
    Matrix approx = L * L.transpose();
    return (A_ - approx).norm();
}