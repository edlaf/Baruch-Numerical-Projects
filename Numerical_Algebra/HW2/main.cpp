#include <iostream>
#include <vector>
#include "../../Helpers/Linear_Algebra_operators/vector.hpp"
#include "../../Helpers/Linear_Algebra_operators/matrix.hpp"
#include "../../Helpers/Linear_Algebra_operators/operators.hpp"
#include "../../Helpers/Linear_system_solvers/Ordinary_least_square.hpp"
#include "../../Helpers/Regressor/Cubic_splines/cubic_splines.hpp"

int main() {
    std::cout << "--- HW2 ---" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "" << std::endl;

    std::cout << "-- Exercice 1 --" << std::endl;
    std::cout << "" << std::endl;

    Vector T2({1.69, 1.81, 1.81, 1.79, 1.79,1.83, 1.81, 1.81, 1.83, 1.81, 1.82, 1.82, 1.80, 1.78, 1.79});
    Vector T3({2.58, 2.71, 2.72, 2.78, 2.77,2.75, 2.71, 2.72, 2.76, 2.73, 2.75, 2.75, 2.73, 2.71, 2.71});
    Vector T5({3.57, 3.69, 3.70, 3.77, 3.77, 3.73, 3.72, 3.74, 3.77, 3.75, 3.77, 3.76, 3.75, 3.72, 3.71});
    Vector T10({4.63, 4.73, 4.74, 4.81, 4.80, 4.79, 4.76, 4.77, 4.80, 4.77, 4.80, 4.80, 4.78, 4.73, 4.73});

    std::cout << "- Question 1" << std::endl;
    Matrix X = Matrix({T2, T5, T10});
    OLS model = Linear_Regression_OLS(X, T3, true);
    model.fit();
    std::cout << "Coefs : " << model.coefs() << std::endl;
    std::cout << "Error : " << model.error(X, T3) << std::endl;

    std::cout << "- Question 2" << std::endl;
    Vector T3_linear_interp = (2.0/3.0)*T2 + (1.0/3.0)*T5;
    std::cout << "Error : " << (T3 - T3_linear_interp).norm(2) << std::endl;

    std::cout << "- Question 3 (natural: s0=0, sn=0 -> natural BC)" << std::endl;
    std::vector<double> t3_nat; t3_nat.reserve(T2.size());
    for (size_t i = 0; i < T2.size(); ++i) {
        Vector xi({2.0, 5.0, 10.0});
        Vector yi({T2[(int)i], T5[(int)i], T10[(int)i]});
        Cubic_spline sp{xi, yi};
        sp.fit(0.0, 0.0);
        t3_nat.push_back(sp.predict(3.0));
    }
    Vector T3_cubic_nat(t3_nat);
    std::cout << "Error : " << (T3 - T3_cubic_nat).norm(2) << std::endl;

    std::cout << "- Question 3 (clamped with secant endpoint slopes)" << std::endl;
    std::vector<double> t3_sec; t3_sec.reserve(T2.size());
    for (size_t i = 0; i < T2.size(); ++i) {
        Vector xi({2.0, 5.0, 10.0});
        Vector yi({T2[(int)i], T5[(int)i], T10[(int)i]});
        double s0 = (yi[1] - yi[0]) / (5.0 - 2.0);
        double sn = (yi[2] - yi[1]) / (10.0 - 5.0);
        Cubic_spline sp{xi, yi};
        sp.fit(s0, sn);
        t3_sec.push_back(sp.predict(3.0));
    }
    Vector T3_cubic_sec(t3_sec);
    std::cout << "Error : " << (T3 - T3_cubic_sec).norm(2) << std::endl;

    std::cout << "- Question 3 (clamped with parabolic endpoint slopes)" << std::endl;
    std::vector<double> t3_par; t3_par.reserve(T2.size());
    for (size_t i = 0; i < T2.size(); ++i) {
        Vector xi({2.0, 5.0, 10.0});
        Vector yi({T2[(int)i], T5[(int)i], T10[(int)i]});
        double y0 = yi[0], y1 = yi[1], y2 = yi[2];
        double s0 = (-11.0/24.0)*y0 + (8.0/15.0)*y1 + (-3.0/40.0)*y2;
        double sn = (  5.0/24.0)*y0 + (-8.0/15.0)*y1 + (13.0/40.0)*y2;
        Cubic_spline sp{xi, yi};
        sp.fit(s0, sn);
        t3_par.push_back(sp.predict(3.0));
    }
    Vector T3_cubic_par(t3_par);
    std::cout << "Error : " << (T3 - T3_cubic_par).norm(2) << std::endl;

    return 0;
}
