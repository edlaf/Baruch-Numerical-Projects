#pragma once
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

/**
 * @class Vector
 * @brief A mathematical 1D vector class with NumPy-like operations.
 *
 * Provides creation methods, element-wise operations, reductions,
 * statistics, broadcasting with scalars, slicing, and boolean masking.
 */
class Vector {
public:
    /** @brief Construct a zero-initialized vector of size n.
     *  @param n Size of the vector.
     */
    explicit Vector(int n);

    /** @brief Construct a vector from a std::vector<double>.
     *  @param n Expected size (must match vect.size()).
     *  @param vect Data used to initialize the vector.
     */
    Vector(int n, const std::vector<double>& vect);

    /** @brief Construct a vector from a std::vector<double>.
     *  @param vect Data used to initialize the vector.
     */
    explicit Vector(const std::vector<double>& vect);

    ~Vector() = default;

    /** @return The number of elements in the vector. */
    size_t size() const;

    /** @brief Print the vector in a readable format. */
    void print() const;

    /** @brief Access an element (modifiable).
     *  @param index Position in the vector.
     *  @return Reference to the element.
     */
    double& operator[](int index);

    /** @brief Access an element (read-only).
     *  @param index Position in the vector.
     *  @return Const reference to the element.
     */
    const double& operator[](int index) const;

    /** @brief Create a zero vector.
     *  @param n Size of the vector.
     *  @return Vector of size n filled with 0.0.
     */
    static Vector zeros(size_t n);

    /** @brief Create a ones vector.
     *  @param n Size of the vector.
     *  @return Vector of size n filled with 1.0.
     */
    static Vector ones(size_t n);

    /** @brief Create a range of values.
     *  @param start First value.
     *  @param stop Last value (excluded).
     *  @param step Increment (default 1.0).
     *  @return Vector containing [start, start+step, ..., < stop].
     */
    static Vector arange(double start, double stop, double step = 1.0);

    /** @brief Create evenly spaced values between start and stop.
     *  @param start First value.
     *  @param stop Last value.
     *  @param num Number of points.
     *  @return Vector with num values from start to stop.
     */
    static Vector linspace(double start, double stop, size_t num);

    /** @brief Create a random vector.
     *  @param n Size of the vector.
     *  @param low Minimum bound.
     *  @param high Maximum bound.
     *  @return Vector with uniform random values in [low, high].
     */
    static Vector random(size_t n, double low = 0.0, double high = 1.0);

    /** @brief Element-wise addition with another vector.
     *  @param other Vector of same size.
     *  @return New vector where res[i] = this[i] + other[i].
     */
    Vector sum(const Vector& other) const;

    /** @brief Element-wise subtraction with another vector.
     *  @param other Vector of same size.
     *  @return New vector where res[i] = this[i] - other[i].
     */
    Vector diff(const Vector& other) const;

    /** @brief Element-wise multiplication with another vector.
     *  @param other Vector of same size.
     *  @return New vector where res[i] = this[i] * other[i].
     */
    Vector multiply(const Vector& other) const;

    /** @brief Element-wise division with another vector.
     *  @param other Vector of same size.
     *  @return New vector where res[i] = this[i] / other[i].
     */
    Vector div(const Vector& other) const;

    /** @brief Element-wise addition with std::vector<T>.
     *  @param other std::vector of same size.
     *  @return New vector where res[i] = this[i] + other[i].
     */
    template<typename T>
    Vector sum(const std::vector<T>& other) const;

    /** @brief Element-wise subtraction with std::vector<T>.
     *  @param other std::vector of same size.
     *  @return New vector where res[i] = this[i] - other[i].
     */
    template<typename T>
    Vector diff(const std::vector<T>& other) const;

    /** @brief Element-wise multiplication with std::vector<T>.
     *  @param other std::vector of same size.
     *  @return New vector where res[i] = this[i] * other[i].
     */
    template<typename T>
    Vector multiply(const std::vector<T>& other) const;

    /** @brief Element-wise division with std::vector<T>.
     *  @param other std::vector of same size.
     *  @return New vector where res[i] = this[i] / other[i].
     */
    template<typename T>
    Vector div(const std::vector<T>& other) const;

    /** @return Sum of all elements. */
    double sum() const;

    /** @return Product of all elements. */
    double prod() const;

    /** @brief Compute cumulative sum in-place. */
    void cumsum();

    /** @brief Compute cumulative product in-place. */
    void cumprod();

    /** @brief Compute differences in-place.
     *  @param d Order of differencing (default 1).
     */
    void diff(int d = 1);

    /** @brief Raise each element to the k-th power.
     *  @param k Exponent (default 2).
     *  @return New vector with pow(this[i], k).
     */
    Vector pow(int k = 2) const;

    /** @brief Compute the L^k norm of the vector.
     *  @param k Norm order (default 2).
     *  @return Norm value.
     */
    double norm(int k = 2) const;

    /** @return Mean value of elements. */
    double mean() const;

    /** @return Variance of elements. */
    double var() const;

    /** @return Standard deviation of elements. */
    double std() const;

    /** @return Minimum element. */
    double min() const;

    /** @return Maximum element. */
    double max() const;

    /** @return Index of minimum element. */
    size_t argmin() const;

    /** @return Index of maximum element. */
    size_t argmax() const;

    /** @brief Append an element to the vector.
     *  @param val Value to add.
     */
    void add(double val);

    /** @brief Same as add().
     *  @param val Value to append.
     */
    void append(double val);

    /** @brief Remove element at index.
     *  @param index Position to remove.
     */
    void remove(int index);

    /** @brief Same as remove().
     *  @param index Position to remove.
     */
    void pop(int index);

    /** @brief Sort vector in ascending order (in-place). */
    void sort();

    /** @brief Return indices that would sort the vector.
     *  @return std::vector of indices.
     */
    std::vector<size_t> argsort() const;

    /** @brief Shift elements by k positions.
     *  @param k Number of positions (positive=right, negative=left).
     */
    void shift(int k);

    /** @brief Slice the vector.
     *  @param start Start index.
     *  @param end End index (exclusive).
     *  @param step Step (default 1).
     *  @return New sliced vector.
     */
    Vector slice(int start, int end, int step = 1) const;

    /** @brief Boolean mask indexing.
     *  @param mask Vector of same size with 0/1.
     *  @return New vector containing elements where mask[i] != 0.
     */
    Vector mask(const Vector& mask) const;

    /** @brief Concatenate another vector at the end.
     *  @param other Vector to append.
     */
    void concat(const Vector& other);

    /** @return True if at least one element != 0. */
    bool any() const;

    /** @return True if all elements != 0. */
    bool all() const;

    /** @return Convert to std::vector<double>. */
    std::vector<double> toStdVector() const;

    /** @brief Element-wise equality comparison. */
    Vector operator==(const Vector& other) const;

    /** @brief Element-wise inequality comparison. */
    Vector operator!=(const Vector& other) const;

    /** @brief Element-wise < comparison. */
    Vector operator<(const Vector& other) const;

    /** @brief Element-wise <= comparison. */
    Vector operator<=(const Vector& other) const;

    /** @brief Element-wise > comparison. */
    Vector operator>(const Vector& other) const;

    /** @brief Element-wise >= comparison. */
    Vector operator>=(const Vector& other) const;

    /** @brief Element-wise addition with another vector. */
    Vector operator+(const Vector& other) const;

    /** @brief Element-wise subtraction with another vector. */
    Vector operator-(const Vector& other) const;

    /** @brief Element-wise multiplication with another vector. */
    Vector operator*(const Vector& other) const;

    /** @brief Element-wise division with another vector. */
    Vector operator/(const Vector& other) const;

    /** @brief Add scalar to each element. */
    Vector operator+(double val) const;

    /** @brief Subtract scalar from each element. */
    Vector operator-(double val) const;

    /** @brief Multiply each element by scalar. */
    Vector operator*(double val) const;

    /** @brief Divide each element by scalar. */
    Vector operator/(double val) const;

    /** @brief Scalar + vector (friend). */
    friend Vector operator+(double val, const Vector& v);

    /** @brief Scalar - vector (friend). */
    friend Vector operator-(double val, const Vector& v);

    /** @brief Scalar * vector (friend). */
    friend Vector operator*(double val, const Vector& v);

    /** @brief Scalar / vector (friend). */
    friend Vector operator/(double val, const Vector& v);

    /** @brief Print operator. */
    friend std::ostream& operator<<(std::ostream& os, const Vector& v);

private:
    std::vector<double> vect_;
};


template<typename T>
Vector Vector::sum(const std::vector<T>& other) const {
    if (size() != other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i = 0; i < size(); i++) res[i] = vect_[i] + static_cast<double>(other[i]);
    return res;
}

template<typename T>
Vector Vector::diff(const std::vector<T>& other) const {
    if (size() != other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i = 0; i < size(); i++) res[i] = vect_[i] - static_cast<double>(other[i]);
    return res;
}

template<typename T>
Vector Vector::multiply(const std::vector<T>& other) const {
    if (size() != other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i = 0; i < size(); i++) res[i] = vect_[i] * static_cast<double>(other[i]);
    return res;
}

template<typename T>
Vector Vector::div(const std::vector<T>& other) const {
    if (size() != other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i = 0; i < size(); i++) {
        if (std::abs(static_cast<double>(other[i])) < 1e-15)
            throw std::runtime_error("Division by zero");
        res[i] = vect_[i] / static_cast<double>(other[i]);
    }
    return res;
}
