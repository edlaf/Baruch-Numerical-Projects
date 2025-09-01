#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include "vector.hpp"

// TODO:
// sparce, diago, rank, inverse, matrice triangulaire
/**
 * @class Matrix
 * @brief A 2D matrix class with operations similar to NumPy.
 *
 * Provides creation methods, element-wise operations, matrix multiplication,
 * and interactions with the Vector class.
 */
class Matrix {
public:
    /**
     * @brief Construct a zero-initialized matrix with given dimensions.
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Matrix(size_t rows, size_t cols);

    /**
     * @brief Construct a matrix from a list of vectors as columns.
     * @param vectors List of vectors, each becoming a column
     */
    Matrix(const std::vector<Vector>& vectors);

    /**
     * @brief Construct a matrix from an initializer list.
     * @param values Nested initializer list (rows of the matrix).
     */
    Matrix(std::initializer_list<std::initializer_list<double>> values);

    /**
     * @brief Construct a matrix from raw 2D data.
     * @param rows Number of rows
     * @param cols Number of columns
     * @param data 2D vector representing matrix entries
     */
    Matrix(size_t rows, size_t cols, const std::vector<std::vector<double>>& data);

    ~Matrix() = default;

    /** @return The number of rows in the matrix. */
    size_t rows() const;

    /** @return The number of columns in the matrix. */
    size_t cols() const;

    /**
     * @brief Access an element by index (modifiable).
     * @param i Row index
     * @param j Column index
     * @return Reference to the element at (i, j)
     */
    double& operator()(size_t i, size_t j);

    /**
     * @brief Access an element by index (read-only).
     * @param i Row index
     * @param j Column index
     * @return Const reference to the element at (i, j)
     */
    const double& operator()(size_t i, size_t j) const;

    /** 
     * @brief Print the matrix in a readable format.
     * @return Nothing
     */
    void print() const;

    /**
     * @brief Create a zero matrix.
     * @param rows Number of rows
     * @param cols Number of columns
     * @return A zero-initialized matrix
     */
    static Matrix zeros(size_t rows, size_t cols);

    /**
     * @brief Create a ones matrix.
     * @param rows Number of rows
     * @param cols Number of columns
     * @return A matrix filled with ones
     */
    static Matrix ones(size_t rows, size_t cols);

    /**
     * @brief Create a random matrix with uniform distribution in [low, high].
     * @param rows Number of rows
     * @param cols Number of columns
     * @param low Lower bound of random values (default 0.0)
     * @param high Upper bound of random values (default 1.0)
     * @return A random matrix
     */
    static Matrix random(size_t rows, size_t cols, double low=0.0, double high=1.0);

    /**
     * @brief Create an identity matrix of size n×n.
     * @param n Dimension
     * @return Identity matrix of size n×n
     */
    static Matrix identity(size_t n);

    /**
     * @brief Element-wise addition with another matrix.
     * @param other The matrix to add
     * @return A new matrix (this + other)
     */
    Matrix operator+(const Matrix& other) const;

    /**
     * @brief Element-wise subtraction with another matrix.
     * @param other The matrix to subtract
     * @return A new matrix (this - other)
     */
    Matrix operator-(const Matrix& other) const;

    /**
     * @brief Element-wise division with another matrix.
     * @param other The matrix to divide by
     * @return A new matrix (this / other)
     */
    Matrix operator/(const Matrix& other) const;

    /**
     * @brief Add a scalar to all elements.
     * @param val The scalar value
     * @return A new matrix where each element is increased by val
     */
    Matrix operator+(double val) const;

    /**
     * @brief Subtract a scalar from all elements.
     * @param val The scalar value
     * @return A new matrix where each element is decreased by val
     */
    Matrix operator-(double val) const;

    /**
     * @brief Multiply all elements by a scalar.
     * @param val The scalar multiplier
     * @return A new matrix where each element is multiplied by val
     */
    Matrix operator*(double val) const;

    /**
     * @brief Divide all elements by a scalar.
     * @param val The scalar divisor
     * @return A new matrix where each element is divided by val
     */
    Matrix operator/(double val) const;

    /**
     * @brief Matrix multiplication (linear algebra).
     * @param other The matrix to multiply with
     * @return The product matrix
     */
    Matrix operator*(const Matrix& other) const;

    /**
     * @brief Multiply a matrix by a vector (A×v).
     * @param v Vector to multiply
     * @return Resulting vector
     */
    Vector operator*(const Vector& v) const;

    /**
     * @brief Multiply a vector by a matrix (v×A).
     * @param v Left vector
     * @param m Right matrix
     * @return A 1×cols matrix
     */
    friend Matrix operator*(const Vector& v, const Matrix& m);

    /**
     * @brief Hadamard product (element-wise multiplication).
     * @param other The matrix to multiply element-wise
     * @return Resulting matrix
     */
    Matrix hadamard(const Matrix& other) const;

    /**
     * @brief Return the transpose of the matrix.
     * @return Transposed matrix
     */
    Matrix transpose() const;

    /**
     * @brief Compute the global L2 norm of the entire matrix.
     * @return sqrt(sum of squares of all entries)
     */
    double norm() const;

        /**
     * @brief Swap two rows in the matrix
     * @param i Index of the first row
     * @param j Index of the second row
     */
    void swapRows(size_t i, size_t j);

    /**
     * @brief Swap two columns in the matrix
     * @param i Index of the first column
     * @param j Index of the second column
     */
    void swapCols(size_t i, size_t j);

    /**
     * @brief Swap two elements in the matrix
     * @param i1 Row index of the first element
     * @param j1 Column index of the first element
     * @param i2 Row index of the second element
     * @param j2 Column index of the second element
     */
    void swapElements(size_t i1, size_t j1, size_t i2, size_t j2);

    /**
     * @brief Create an identity matrix of size n×n (alias for identity)
     * @param n Number of rows/columns
     * @return Identity matrix
     */
    static Matrix eye(size_t n);

    /**
     * @brief test wether a matrix is up or down triangular
     * @param M matrix
     * @param up bool (true to cheak if it's a up triangular)
     * @return bool = true if up (with up = true)
     */
    static bool is_triangular(const Matrix& M, bool up = true);

    /**
     * @brief Retourne une nouvelle matrice avec une colonne de 1 ajoutée au début.
     *
     * Exemple :
     *  X = [[x11, x12],
     *       [x21, x22]]
     *
     *  X.add_intercept_column() =
     *      [[1, x11, x12],
     *       [1, x21, x22]]
     *
     * @return Matrix avec une colonne de 1 en plus.
     */
    Matrix add_intercept_column() const;
private:
    size_t rows_, cols_;
    std::vector<std::vector<double>> data_;
};
