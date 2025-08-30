#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include "../Linear_Algebra_operators/matrix.hpp"
#include "../Linear_Algebra_operators/vector.hpp"

class Cholesky{

public:

    Cholesky(Matrix A);

    double verify_Cholesky();
    Matrix decompose();

private:
    Matrix A_;
    Matrix L;
    size_t n_;
};
