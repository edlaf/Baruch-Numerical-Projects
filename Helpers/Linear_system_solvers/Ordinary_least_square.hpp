#pragma once
#include <iostream>
#include "../Linear_Algebra_operators/matrix.hpp"
#include "../Linear_Algebra_operators/vector.hpp"
#include "../Cholesky/Cholesky.hpp"
#include "../Linear_system_solvers/Linear_solver.hpp"

/**
 * @class OLS
 * @brief Implements Ordinary Least Squares (OLS) linear regression.
 *
 * This class solves the normal equation (XᵀX)β = Xᵀy using Cholesky decomposition.
 * It allows fitting a linear model, retrieving coefficients, making predictions,
 * and evaluating performance with MSE and R².
 */
class OLS {
public:
    /**
     * @brief Constructor for the OLS class.
     *
     * @param X Design matrix (features). If add_intercept = true, a column of ones
     *          will be added automatically as the intercept.
     * @param y Target vector (observed values).
     * @param add_intercept If true (default), automatically adds a column of ones for the intercept term.
     */
    OLS(Matrix X, Vector y, bool add_intercept = true);

    /**
     * @brief Fits the OLS model by estimating coefficients β.
     *
     * Solves (XᵀX)β = Xᵀy using Cholesky decomposition.
     *
     * @throws std::runtime_error if the decomposition fails.
     */
    void fit();

    /**
     * @brief Predicts output values for a given feature matrix X.
     *
     * @param X Feature matrix for prediction.
     * @return Vector of predicted values.
     *
     * @throws std::runtime_error if the model has not been fitted yet.
     */
    Vector predict(const Matrix& X) const;

    /**
     * @brief Predicts a single output value for a given feature vector x.
     *
     * @param x Feature vector representing one observation.
     * @return Predicted scalar value.
     *
     * @throws std::runtime_error if the model has not been fitted yet.
     */
    double predict(const Vector& x) const;

    /**
     * @brief Returns the fitted model coefficients β.
     *
     * @return Vector of coefficients (intercept included if add_intercept was set to true).
     *
     * @throws std::runtime_error if the model has not been fitted yet.
     */
    Vector coefs() const;

    /**
     * @brief Computes the Mean Squared Error (MSE).
     *
     * @param X Feature matrix.
     * @param y Ground truth target values.
     * @return Double representing the mean squared error.
     *
     * @throws std::runtime_error if the model has not been fitted or if dimensions mismatch.
     */
    double mse(const Matrix& X, const Vector& y) const;

    /**
     * @brief Computes the coefficient of determination R².
     *
     * @param X Feature matrix.
     * @param y Ground truth target values.
     * @return Double representing the R² score (between -∞ and 1).
     *
     * @throws std::runtime_error if the model has not been fitted or if dimensions mismatch.
     */
    double r2(const Matrix& X, const Vector& y) const;


    /**
     * @brief Computes the Mean Squared Error (MSE).
     *
     * @param X Feature matrix.
     * @param y Ground truth target values.
     * @return Double representing the mean squared error.
     *
     * @throws std::runtime_error if the model has not been fitted or if dimensions mismatch.
     */
    double error(const Matrix& X, const Vector& y) const;
private:
    Vector y_;
    bool fitted_;
    bool add_intercept_;
    Matrix X_;
    Vector beta;
};

