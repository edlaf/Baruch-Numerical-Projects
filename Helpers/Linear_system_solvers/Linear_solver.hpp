#pragma once
#include <iostream>
#include "../Linear_Algebra_operators/matrix.hpp"
#include "../Linear_Algebra_operators/vector.hpp"
#include "../Linear_Algebra_operators/operators.hpp"
#include "../Cholesky/Cholesky.hpp"

class Linear_Solver{

public:
    Linear_Solver(Matrix A, Vector vect);

    Vector Cholesky_solver() const;
    Vector LU_solver() const;

    Vector Diagonal_solver() const;

    Vector Triangular_solver() const;

private:
    Matrix A_;
    Vector vect_;
};