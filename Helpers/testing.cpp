#include "../Linear_Algebra_operators/vector.hpp"
#include "../Linear_Algebra_operators/matrix.hpp"
#include "../Linear_Algebra_operators/operators.hpp"
#include "../Cholesky/Cholesky.hpp"
#include <iostream>

int main() {
    Matrix A({{4, 2},{2, 3}});
    std::cout << "Matrix A:" << std::endl;
    A.print();

    Matrix L = Cholesky_decompose(A);
    std::cout << "\nMatrix L (Cholesky factor):" << std::endl;
    L.print();

    Matrix approx = L * L.transpose();
    std::cout << "\nReconstructed A = L * L^T:" << std::endl;
    approx.print();

    double error = (A - approx).norm();
    std::cout << "\nVerification error ||A - L*L^T|| = " << error << std::endl;

    return 0;
}
