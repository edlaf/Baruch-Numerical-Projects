#include "cubic_splines.hpp"
#include <vector>

Cubic_spline::Cubic_spline(Vector points, Vector values)
    : x_(points), y_(values), n_(points.size()),
      h_(std::max<std::size_t>(1, points.size() > 0 ? points.size() - 1 : 1)),
      M_(points.size()),
      a_(std::max<std::size_t>(1, points.size() > 0 ? points.size() - 1 : 1)),
      b_(std::max<std::size_t>(1, points.size() > 0 ? points.size() - 1 : 1)),
      c_(std::max<std::size_t>(1, points.size() > 0 ? points.size() - 1 : 1)),
      d_(std::max<std::size_t>(1, points.size() > 0 ? points.size() - 1 : 1)),
      fitted_(false)
{
    if (x_.size() != y_.size()) throw std::runtime_error("Cubic_spline: points and values must have the same size");
    if (n_ < 2) throw std::runtime_error("Cubic_spline: at least two points are required");
    for (std::size_t i = 0; i + 1 < n_; ++i) {
        double dx = x_[static_cast<int>(i + 1)] - x_[static_cast<int>(i)];
        if (dx <= 0.0) throw std::runtime_error("Cubic_spline: x must be strictly increasing");
        h_[static_cast<int>(i)] = dx;
    }
}

void Cubic_spline::fit(double s0, double sn) {
    std::size_t n = n_;
    std::vector<double> lower(n, 0.0), diag(n, 0.0), upper(n, 0.0), rhs(n, 0.0);
    bool natural = (s0 == 0.0 && sn == 0.0);

    if (natural) {
        diag[0] = 1.0; rhs[0] = 0.0;
        for (std::size_t i = 1; i + 1 < n; ++i) {
            double him1 = h_[static_cast<int>(i - 1)];
            double hi   = h_[static_cast<int>(i)];
            lower[i] = him1;
            diag[i]  = 2.0 * (him1 + hi);
            upper[i] = hi;
            double term1 = (y_[static_cast<int>(i + 1)] - y_[static_cast<int>(i)]) / hi;
            double term2 = (y_[static_cast<int>(i)]     - y_[static_cast<int>(i - 1)]) / him1;
            rhs[i] = 6.0 * (term1 - term2);
        }
        diag[n - 1] = 1.0; rhs[n - 1] = 0.0;
    } else {
        diag[0] = 2.0 * h_[0];
        upper[0] = h_[0];
        rhs[0]   = 6.0 * ((y_[1] - y_[0]) / h_[0] - s0);

        for (std::size_t i = 1; i + 1 < n; ++i) {
            double him1 = h_[static_cast<int>(i - 1)];
            double hi   = h_[static_cast<int>(i)];
            lower[i] = him1;
            diag[i]  = 2.0 * (him1 + hi);
            upper[i] = hi;
            double term1 = (y_[static_cast<int>(i + 1)] - y_[static_cast<int>(i)]) / hi;
            double term2 = (y_[static_cast<int>(i)]     - y_[static_cast<int>(i - 1)]) / him1;
            rhs[i] = 6.0 * (term1 - term2);
        }

        lower[n - 1] = h_[n - 2];
        diag[n - 1]  = 2.0 * h_[n - 2];
        rhs[n - 1]   = 6.0 * (sn - (y_[n - 1] - y_[n - 2]) / h_[n - 2]);
    }

    for (std::size_t i = 1; i < n; ++i) {
        double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i]  -= w * rhs[i - 1];
    }

    std::vector<double> Mv(n, 0.0);
    Mv[n - 1] = rhs[n - 1] / diag[n - 1];
    for (int i = static_cast<int>(n) - 2; i >= 0; --i) {
        Mv[i] = (rhs[i] - upper[i] * Mv[i + 1]) / diag[i];
    }
    for (std::size_t i = 0; i < n; ++i) M_[static_cast<int>(i)] = Mv[i];

    for (std::size_t i = 0; i + 1 < n; ++i) {
        double hi  = h_[static_cast<int>(i)];
        double yi  = y_[static_cast<int>(i)];
        double yi1 = y_[static_cast<int>(i + 1)];
        double Mi  = M_[static_cast<int>(i)];
        double Mi1 = M_[static_cast<int>(i + 1)];
        a_[static_cast<int>(i)] = yi;
        b_[static_cast<int>(i)] = (yi1 - yi) / hi - hi * (2.0 * Mi + Mi1) / 6.0;
        c_[static_cast<int>(i)] = Mi / 2.0;
        d_[static_cast<int>(i)] = (Mi1 - Mi) / (6.0 * hi);
    }

    fitted_ = true;
}

int Cubic_spline::find_interval(double x) const {
    if (x <= x_[0]) return 0;
    if (x >= x_[static_cast<int>(n_ - 1)]) return static_cast<int>(n_ - 2);
    for (std::size_t i = 0; i + 1 < n_; ++i) {
        if (x >= x_[static_cast<int>(i)] && x <= x_[static_cast<int>(i + 1)])
            return static_cast<int>(i);
    }
    return static_cast<int>(n_ - 2);
}

double Cubic_spline::predict(double x) const {
    if (!fitted_) throw std::runtime_error("Cubic_spline::predict: call fit() before prediction");
    int i = find_interval(x);
    double dx = x - x_[i];
    double ai = a_[i], bi = b_[i], ci = c_[i], di = d_[i];
    return ((di * dx + ci) * dx + bi) * dx + ai;
}

Vector Cubic_spline::predict(const Vector& xs) const {
    if (!fitted_) throw std::runtime_error("Cubic_spline::predict: call fit() before prediction");
    std::vector<double> out;
    out.reserve(xs.size());
    for (std::size_t k = 0; k < xs.size(); ++k) {
        double x = xs[static_cast<int>(k)];
        out.push_back(predict(x));
    }
    return Vector(out);
}
