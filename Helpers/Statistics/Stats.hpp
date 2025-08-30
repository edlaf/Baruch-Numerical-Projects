#pragma once
#include <iostream>
#include "../Linear_Algebra_operators/matrix.cpp"
#include "../Linear_Algebra_operators/vector.cpp"

/**
 * @brief Compute the arithmetic mean (average) of all elements in the vector.
 * @param vect Input Vector
 * @return The mean value as a double
 */
double mean(const Vector& vect);

/**
 * @brief Compute the standard deviation of all elements in the vector.
 * It measures the spread of the values around the mean.
 * @param vect Input Vector
 * @return The standard deviation as a double
 */
double std(const Vector& vect);

/**
 * @brief Compute the variance of all elements in the vector.
 * It is the square of the standard deviation.
 * @param vect Input Vector
 * @return The variance as a double
 */
double var(const Vector& vect);

/**
 * @brief Compute the rolling (moving) mean of the vector using a fixed window size.
 * For each position i >= k-1, the result is the mean of the last k values.
 * Positions before that can be set to NaN or 0 depending on convention.
 * @param vect Input Vector
 * @param k Window size
 * @return A Vector of rolling means of the same size as input
 */
Vector rolling_mean(const Vector& vect, int k);

/**
 * @brief Compute the rolling (moving) standard deviation of the vector using a fixed window size.
 * For each position i >= k-1, the result is the standard deviation of the last k values.
 * Positions before that can be set to NaN or 0 depending on convention.
 * @param vect Input Vector
 * @param k Window size
 * @return A Vector of rolling standard deviations of the same size as input
 */
Vector rolling_std(const Vector& vect, int k);

/**
 * @brief Compute the rolling (moving) variance of the vector using a fixed window size.
 * For each position i >= k-1, the result is the variance of the last k values.
 * Positions before that can be set to NaN or 0 depending on convention.
 * @param vect Input Vector
 * @param k Window size
 * @return A Vector of rolling variances of the same size as input
 */
Vector rolling_var(const Vector& vect, int k);

/**
 * @brief Compute the covariance matrix of a dataset.
 * @param data Matrix of size (n_samples x n_features) where each row is a data point.
 * @return Covariance matrix of size (n_features x n_features).
 */
Matrix covariance(const Matrix& data);