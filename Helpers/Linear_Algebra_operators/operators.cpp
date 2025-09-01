#include "operators.hpp"
#include "../Linear_system_solvers/Ordinary_least_square.hpp"

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

Vector system_solver(const Matrix& A, const Vector& b, const std::string& method) {
    Linear_Solver solver(A, b);

    if (method == "diagonal") {
        return solver.Diagonal_solver();
    } 
    else if (method == "triangular") {
        return solver.Triangular_solver();
    } 
    else if (method == "cholesky") {
        return solver.Cholesky_solver();
    } 
    else if (method == "lu") {
        return solver.LU_solver();
    } 
    else {
        throw std::runtime_error("Méthode inconnue : " + method);
    }
}

OLS Linear_Regression_OLS(const Matrix& X, const Vector& y, bool add_intercept) {
    return OLS(X, y, add_intercept);
}
