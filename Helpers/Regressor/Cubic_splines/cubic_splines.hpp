#pragma once
#include <stdexcept>
#include <cstddef>
#include <algorithm>
#include "../../Linear_Algebra_operators/vector.hpp"
#include "../../Linear_Algebra_operators/matrix.hpp"
#include "../../Linear_Algebra_operators/operators.hpp"

/**
 * @brief Cubic spline interpolator with selectable boundary:
 *        if s0 == 0 and sn == 0 -> natural (M0 = M_{n-1} = 0),
 *        else -> clamped with S'(x0)=s0, S'(x_{n-1})=sn.
 */
class Cubic_spline {
public:
    /**
     * @brief Construct a spline from nodes and values.
     * @param points Strictly increasing x-coordinates (size n >= 2).
     * @param values Function values y at the given x (same size as points).
     */
    Cubic_spline(Vector points, Vector values);

    /**
     * @brief Fit the spline. Defaults reproduce natural boundary.
     * @param s0 Left-end slope S'(x0). If both s0 and sn are 0.0, uses natural BC.
     * @param sn Right-end slope S'(x_{n-1}). If both s0 and sn are 0.0, uses natural BC.
     */
    void fit(double s0 = 0.0, double sn = 0.0);

    /**
     * @brief Evaluate the spline at a single x.
     */
    double predict(double x) const;

    /**
     * @brief Evaluate the spline on a vector of x.
     */
    Vector predict(const Vector& xs) const;

private:
    int find_interval(double x) const;

    Vector x_;
    Vector y_;
    std::size_t n_;
    Vector h_;
    Vector M_;
    Vector a_;
    Vector b_;
    Vector c_;
    Vector d_;
    bool fitted_;
};
