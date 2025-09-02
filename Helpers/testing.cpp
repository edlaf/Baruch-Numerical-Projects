#include "../Linear_Algebra_operators/vector.hpp"
#include "../Linear_Algebra_operators/matrix.hpp"
#include "../Linear_Algebra_operators/operators.hpp"
#include "../Cholesky/Cholesky.hpp"
#include "../Linear_system_solvers/Linear_solver.hpp"
#include "../Linear_system_solvers/Ordinary_least_square.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "../Regressor/Cubic_splines/cubic_splines.hpp"

static constexpr double PI = 3.141592653589793238462643383279502884;

int main() {
    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(8);

    std::vector<double> xs;
    for (int i = 0; i <= 16; ++i) xs.push_back((2.0 * PI) * i / 16.0);

    std::vector<double> ys(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) ys[i] = std::sin(xs[i]);

    Cubic_spline sp{ Vector(xs), Vector(ys) };
    sp.fit();

    bool pass_knots = true;
    for (size_t i = 0; i < xs.size(); ++i) {
        double yhat = sp.predict(xs[i]);
        if (std::abs(yhat - ys[i]) > 1e-10) {
            pass_knots = false;
            break;
        }
    }
    std::cout << "[Test 1] Interpolation at knots: " << (pass_knots ? "PASS" : "FAIL") << "\n";

    std::vector<double> xq;
    for (int i = 0; i <= 400; ++i) xq.push_back((2.0 * PI) * i / 400.0);
    Vector yq = sp.predict(Vector(xq));

    double max_err = 0.0;
    for (size_t i = 0; i < xq.size(); ++i) {
        double err = std::abs(yq[(int)i] - std::sin(xq[i]));
        if (err > max_err) max_err = err;
    }
    std::cout << "[Test 1] Max |error| vs sin(x) on [0,2π]: " << max_err << "\n";

    auto num_deriv = [&](double x){
        const double eps = 1e-6;
        return (sp.predict(x + eps) - sp.predict(x - eps)) / (2.0 * eps);
    };
    double max_jump = 0.0;
    for (size_t i = 1; i + 1 < xs.size(); ++i) {
        double dl = num_deriv(xs[i] - 1e-6);
        double dr = num_deriv(xs[i] + 1e-6);
        max_jump = std::max(max_jump, std::abs(dl - dr));
    }
    std::cout << "[Test 1] Max |S'(x_i^-)-S'(x_i^+)|: " << max_jump << "\n";

    std::vector<double> xu {0.0, 0.15, 0.35, 0.9, 1.7, 2.05, 3.4, 4.1, 5.0, 6.28};
    std::vector<double> yu(xu.size());
    for (size_t i = 0; i < xu.size(); ++i) yu[i] = std::sin(xu[i]);

    Cubic_spline sp2{ Vector(xu), Vector(yu) };
    sp2.fit();

    bool pass_knots2 = true;
    for (size_t i = 0; i < xu.size(); ++i) {
        double yhat = sp2.predict(xu[i]);
        if (std::abs(yhat - yu[i]) > 1e-10) {
            pass_knots2 = false;
            break;
        }
    }
    std::cout << "[Test 2] Interpolation at knots (non-uniform): " << (pass_knots2 ? "PASS" : "FAIL") << "\n";

    std::vector<double> xq2;
    for (int i = 0; i <= 400; ++i) xq2.push_back((2.0 * PI) * i / 400.0);
    Vector yq2 = sp2.predict(Vector(xq2));

    double max_err2 = 0.0;
    for (size_t i = 0; i < xq2.size(); ++i) {
        double err = std::abs(yq2[(int)i] - std::sin(xq2[i]));
        if (err > max_err2) max_err2 = err;
    }
    std::cout << "[Test 2] Max |error| vs sin(x) on [0,2π]: " << max_err2 << "\n";

    double left_ex = sp.predict(xs.front() - 0.5);
    double right_ex = sp.predict(xs.back() + 0.5);
    std::cout << "[Test 3] Extrapolation: S(x0-0.5)=" << left_ex
            << ", S(xn+0.5)=" << right_ex << "\n";

    return 0;
}
