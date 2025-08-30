#include "Stats.hpp"

double mean(const Vector& vect) {
    return vect.mean();
}

double std(const Vector& vect) {
    return vect.std();
}

double var(const Vector& vect) {
    return vect.var();
}

Vector rolling_mean(const Vector& vect, int k) {
    if (k <= 0 || k > (int)vect.size()) throw std::runtime_error("Invalid window size for rolling_mean");
    Vector res(std::vector<double>(vect.size(), NAN));
    for (size_t i = k - 1; i < vect.size(); i++) {
        Vector window = vect.slice(i + 1 - k, i + 1);
        res[i] = window.mean();
    }
    return res;
}

Vector rolling_var(const Vector& vect, int k) {
    if (k <= 0 || k > (int)vect.size()) throw std::runtime_error("Invalid window size for rolling_var");
    Vector res(std::vector<double>(vect.size(), NAN));
    for (size_t i = k - 1; i < vect.size(); i++) {
        Vector window = vect.slice(i + 1 - k, i + 1);
        res[i] = window.var();
    }
    return res;
}

Vector rolling_std(const Vector& vect, int k) {
    if (k <= 0 || k > (int)vect.size()) throw std::runtime_error("Invalid window size for rolling_std");
    Vector res(std::vector<double>(vect.size(), NAN));
    for (size_t i = k - 1; i < vect.size(); i++) {
        Vector window = vect.slice(i + 1 - k, i + 1);
        res[i] = window.std();
    }
    return res;
}

Matrix covariance(const Matrix& data) {
    size_t n = data.rows();
    size_t d = data.cols();

    if (n < 2) throw std::runtime_error("Not enough data points for covariance");

    std::vector<double> means(d, 0.0);
    for (size_t j = 0; j < d; j++) {
        for (size_t i = 0; i < n; i++) {
            means[j] += data(i, j);
        }
        means[j] /= n;
    }

    Matrix cov(d, d);
    for (size_t i = 0; i < d; i++) {
        for (size_t j = i; j < d; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < n; k++) {
                sum += (data(k, i) - means[i]) * (data(k, j) - means[j]);
            }
            cov(i, j) = sum / (n - 1);
            cov(j, i) = cov(i, j);
        }
    }

    return cov;
}

