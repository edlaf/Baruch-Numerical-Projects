#include "Ordinary_least_square.hpp"
#include <stdexcept>

OLS::OLS(Matrix X, Vector y, bool add_intercept)
    : y_(y), fitted_(false), add_intercept_(add_intercept), X_(add_intercept ? X.add_intercept_column() : X), beta(Vector::zeros(add_intercept ? X.cols() + 1 : X.cols())){}

void OLS::fit() {
    Matrix Xt = X_.transpose();
    Matrix XtX = Xt * X_;
    Vector Xty = Xt * y_;

    Linear_Solver solver(XtX, Xty);
    beta = solver.Cholesky_solver();

    fitted_ = true;
}

Vector OLS::predict(const Matrix& X) const {
    if (!fitted_) {
        throw std::runtime_error("OLS::predict before fit()");
    }

    Matrix Xused = add_intercept_ ? X.add_intercept_column() : X;
    return Xused * beta;
}

double OLS::predict(const Vector& x) const {
    if (!fitted_) {
        throw std::runtime_error("OLS::predict before fit()");
    }

    if (add_intercept_) {
        Vector xnew(x.size() + 1);
        xnew[0] = 1.0;
        for (size_t i = 0; i < x.size(); ++i) {
            xnew[i + 1] = x[i];
        }
        return xnew.dot(beta);
    } else {
        return x.dot(beta);
    }
}


Vector OLS::coefs() const {
    if (!fitted_) {
        throw std::runtime_error("OLS::coefs before fit()");
    }
    return beta;
}

double OLS::mse(const Matrix& X, const Vector& y) const {
    if (!fitted_) throw std::runtime_error("OLS::mse before fit()");
    if (X.rows() != y.size()) throw std::runtime_error("OLS::mse dimension mismatch");

    Vector y_pred = predict(X);
    double sum_sq = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        double diff = y[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    return sum_sq / y.size();
}

double OLS::r2(const Matrix& X, const Vector& y) const {
    if (!fitted_) throw std::runtime_error("OLS::r2 before fit()");
    if (X.rows() != y.size()) throw std::runtime_error("OLS::r2 dimension mismatch");

    Vector y_pred = predict(X);

    double mean_y = y.mean();

    double ss_res = 0.0;
    double ss_tot = 0.0;

    for (size_t i = 0; i < y.size(); i++) {
        double diff_res = y[i] - y_pred[i];
        double diff_tot = y[i] - mean_y;
        ss_res += diff_res * diff_res;
        ss_tot += diff_tot * diff_tot;
    }

    return 1.0 - (ss_res / ss_tot);
}

double OLS::error(const Matrix& X, const Vector& y) const {
    if (!fitted_) throw std::runtime_error("OLS::mse before fit()");
    if (X.rows() != y.size()) throw std::runtime_error("OLS::mse dimension mismatch");

    Vector y_pred = predict(X);
    double sum_sq = 0.0;
    for (size_t i = 0; i < y.size(); i++) {
        double diff = y[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq);
}