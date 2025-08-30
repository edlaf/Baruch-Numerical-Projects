#pragma once
#include "matrix.hpp"
#include "vector.hpp"
#include "../Cholesky/Cholesky.hpp"
#include "../LU_decomposition/LU_decomposition.hpp"
#include "../Inverse/Inverse.hpp"

/**
 * @brief Apply LU decomposition on a matrix
 */
Matrix LU_decompose(const Matrix& A);

/**
 * @brief Apply Cholesky decomposition on a matrix
 */
Matrix Cholesky_decompose(const Matrix& A);

/**
 * @brief Compute inverse of a matrix
 */
Matrix inverse(const Matrix& A);
