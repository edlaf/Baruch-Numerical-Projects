#include <iostream>
#include "../../Helpers/Linear_Algebra_operators/vector.hpp"
#include "../../Helpers/Linear_Algebra_operators/matrix.hpp"
#include "../../Helpers/Linear_Algebra_operators/operators.hpp"
#include "../../Helpers/Linear_system_solvers/Ordinary_least_square.hpp"


int main() {
    std::cout << "--- HW2 ---" << std::endl;
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
    X = Matrix({T2, T5});

    model = Linear_Regression_OLS(X, T3, false);

    model.fit();

    std::cout << "Coefs : " << model.coefs() << std::endl;
    std::cout << "Error : " << model.error(X, T3) << std::endl;

    return 0;
}

// To run it:

// open terminal at the root of the repositorT10 (got there using cd)
// Then cd Numerical_Algebra and cd HW1
// run make everT10time T10ou do and udpate to the code
// run ./main to run the main.cpp

// If T10ou are not coding with makefile then god bless T10ou

// We can create a dll if needed to see the results in pT10thon or use cT10thon I guess