#pragma once
#include "matrix.hpp"
#include "vector.hpp"
#include "../Cholesky/Cholesky.hpp"
#include "../LU_decomposition/LU_decomposition.hpp"
#include "../Inverse/Inverse.hpp"
#include "../Linear_system_solvers/Linear_solver.hpp"

/**
 * @brief Apply LU decomposition on a matrix.
 *
 * Factorizes the matrix A into L (lower triangular) and U (upper triangular),
 * such that A = L * U.
 *
 * @param A Input square matrix (must be non-singular).
 * @return A Matrix that encodes the LU decomposition (implementation-dependent).
 */
std::vector<Matrix> LU_decompose_with_pivot(const Matrix& A);

/**
 * @brief Apply LU decomposition on a matrix.
 *
 * Factorizes the matrix A into L (lower triangular) and U (upper triangular),
 * such that A = L * U.
 *
 * @param A Input square matrix (must be non-singular).
 * @return A Matrix that encodes the LU decomposition (implementation-dependent).
 */
std::vector<Matrix> LU_decompose_without_pivot(const Matrix& A);

/**
 * @brief Apply Cholesky decomposition on a matrix.
 *
 * Factorizes the matrix A into L (lower triangular),
 * such that A = L * L^T. Requires A to be symmetric positive definite (SPD).
 *
 * @param A Input square SPD matrix.
 * @return Lower triangular matrix L.
 */
Matrix Cholesky_decompose(const Matrix& A);

/**
 * @brief Compute the inverse of a matrix.
 *
 * Uses Gaussian elimination, LU decomposition, or other methods depending
 * on implementation. The matrix must be non-singular.
 *
 * @param A Input square matrix.
 * @return Inverse matrix A⁻¹.
 */
Matrix inverse(const Matrix& A);

/**
 * @brief Solve a linear system Ax = b using a specified method.
 *
 * Available methods:
 * - "diagonal"  : assumes A is diagonal
 * - "triangular": assumes A is triangular (upper or lower)
 * - "lu"        : LU decomposition with pivoting (default)
 * - "cholesky"  : Cholesky decomposition (requires SPD matrix)
 *
 * @param A Input square matrix.
 * @param b Right-hand side vector.
 * @param method Solving method (default = "lu").
 * @return Solution vector x such that A*x ≈ b.
 *
 * @throws std::runtime_error if the method is unknown or A is not suitable.
 */
Vector system_solver(const Matrix& A, const Vector& b, const std::string& method = "lu");
