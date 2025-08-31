#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include "../Linear_Algebra_operators/matrix.hpp"

class LU{

public:

    LU(Matrix A);

    double verify_();
    std::vector<Matrix> decompose_without_pivot();
    std::vector<Matrix> decompose_with_pivot();

private:
    Matrix A_;
    Matrix L;
    Matrix U;
    Matrix P;
    size_t n_;
};
