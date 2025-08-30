#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include "vector.hpp"

/**
 * @class Matrix
 * @brief A 2D matrix class with operations similar to NumPy.
 *
 * Provides creation methods, element-wise operations, matrix multiplication,
 * and interactions with the Vector class.
 */
class Matrix {
public:
    /** @brief Construct a zero-initialized matrix with given dimensions. */
    Matrix(size_t rows, size_t cols);

    /** @brief Construct a matrix from given data. */
    Matrix(size_t rows, size_t cols, const std::vector<std::vector<double>>& data);

    ~Matrix() = default;

    /** @return The number of rows in the matrix. */
    size_t rows() const;

    /** @return The number of columns in the matrix. */
    size_t cols() const;

    /** @brief Access an element by index (modifiable). */
    double& operator()(size_t i, size_t j);

    /** @brief Access an element by index (read-only). */
    const double& operator()(size_t i, size_t j) const;

    /** @brief Print the matrix in a readable format. */
    void print() const;

    /** @brief Create a zero matrix. */
    static Matrix zeros(size_t rows, size_t cols);

    /** @brief Create a ones matrix. */
    static Matrix ones(size_t rows, size_t cols);

    /** @brief Create a random matrix with uniform distribution in [low, high]. */
    static Matrix random(size_t rows, size_t cols, double low=0.0, double high=1.0);

    /** @brief Create an identity matrix of size n×n. */
    static Matrix identity(size_t n);

    /** @brief Element-wise addition with another matrix. */
    Matrix operator+(const Matrix& other) const;

    /** @brief Element-wise subtraction with another matrix. */
    Matrix operator-(const Matrix& other) const;

    /** @brief Element-wise division with another matrix. */
    Matrix operator/(const Matrix& other) const;

    /** @brief Add a scalar to all elements. */
    Matrix operator+(double val) const;

    /** @brief Subtract a scalar from all elements. */
    Matrix operator-(double val) const;

    /** @brief Multiply all elements by a scalar. */
    Matrix operator*(double val) const;

    /** @brief Divide all elements by a scalar. */
    Matrix operator/(double val) const;

    /** @brief Matrix multiplication (linear algebra). */
    Matrix operator*(const Matrix& other) const;

    /** @brief Multiply a matrix by a vector (A×v). */
    Vector operator*(const Vector& v) const;

    /** @brief Multiply a vector by a matrix (v×A). */
    friend Matrix operator*(const Vector& v, const Matrix& m);

    /** @brief Hadamard product (element-wise multiplication). */
    Matrix hadamard(const Matrix& other) const;

    /** @brief Return the transpose of the matrix. */
    Matrix transpose() const;

private:
    size_t rows_, cols_;
    std::vector<std::vector<double>> data_;
};
