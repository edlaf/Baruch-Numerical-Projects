#include "Linear_solver.hpp"

Linear_Solver::Linear_Solver(Matrix A, Vector vect):A_(A), vect_(vect){}

Vector Linear_Solver::Cholesky_solver() const {
    if (A_.rows() != A_.cols()) {
        throw std::runtime_error("La matrice n'est pas carrée.");
    }

    Matrix L = Cholesky_decompose(A_);
    size_t n = A_.rows();

    Vector y(n), x(n);

    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L(i, j) * y[j];
        }
        y[i] = (vect_[i] - sum) / L(i, i);
    }

    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += L(j, i) * x[j];
        }
        x[i] = (y[i] - sum) / L(i, i);
    }

    return x;
}


Vector Linear_Solver::LU_solver() const {
    if (A_.rows() != A_.cols()) {
        throw std::runtime_error("La matrice n'est pas carrée.");
    }

    auto LU_pivot = LU_decompose_with_pivot(A_);
    Matrix P = LU_pivot[0];
    Matrix L = LU_pivot[1];
    Matrix U = LU_pivot[2];
    size_t n = A_.rows();

    Vector Pb = P * vect_;

    Vector y(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L(i, j) * y[j];
        }
        if (std::abs(L(i, i)) < 1e-13) {
            throw std::runtime_error("Pivot nul rencontré dans LU (forward).");
        }
        y[i] = (Pb[i] - sum) / L(i, i);
    }

    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            sum += U(i, j) * x[j];
        }
        if (std::abs(U(i, i)) < 1e-13) {
            throw std::runtime_error("Pivot nul rencontré dans LU (backward).");
        }
        x[i] = (y[i] - sum) / U(i, i);
    }

    return x;
}


Vector Linear_Solver::Diagonal_solver() const{
    Vector solution = Vector(A_.cols());
    for (size_t i = 0; i < Vector::len(solution); i++){
        solution[i] = vect_[i] / A_(i,i);
    }
    return solution;
}

Vector Linear_Solver::Triangular_solver() const {
    if (A_.rows() != A_.cols()) {
        throw std::runtime_error("La matrice n'est pas carrée.");
    }

    bool up = Matrix::is_triangular(A_, true);
    bool low = Matrix::is_triangular(A_, false);

    if (!up && !low) {
        throw std::runtime_error("La matrice n'est pas triangulaire.");
    }

    size_t n = A_.rows();
    Vector x(n);

    if (up) {
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (size_t j = i + 1; j < n; ++j) {
                sum += A_(i, j) * x[j];
            }
            if (std::abs(A_(i, i)) < 1e-13) {
                throw std::runtime_error("Pivot nul rencontré.");
            }
            x[i] = (vect_[i] - sum) / A_(i, i);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += A_(i, j) * x[j];
            }
            if (std::abs(A_(i, i)) < 1e-13) {
                throw std::runtime_error("Pivot nul rencontré.");
            }
            x[i] = (vect_[i] - sum) / A_(i, i);
        }
    }

    return x;
}
