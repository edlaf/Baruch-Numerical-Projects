#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include "../Linear_Algebra_operators/matrix.hpp"

class LU{

public:

    LU(Matrix A);

    Matrix test();
    Matrix decompose();

private:
    Matrix A_;
};
