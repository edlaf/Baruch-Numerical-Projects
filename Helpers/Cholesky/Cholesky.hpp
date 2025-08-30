#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include "../Linear_Algebra_operators/matrix.hpp"
#include "../Linear_Algebra_operators/vector.hpp"

class Cholesky{

public:

    Cholesky(Matrix A);

    Matrix test();
    Matrix decompose();

private:
    Matrix A_;
};
