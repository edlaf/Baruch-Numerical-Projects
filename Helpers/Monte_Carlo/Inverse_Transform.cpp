#include <bits/stdc++.h>
using namespace std;

static inline double inv_norm_cdf_BSM(double u) {
    if (u <= 0.0) return -INFINITY;
    if (u >= 1.0) return  INFINITY;

    static const double a0 =  2.50662823884;
    static const double a1 = -18.61500062529;
    static const double a2 =  41.39119773534;
    static const double a3 = -25.44106049637;

    static const double b0 = -8.47351093090;
    static const double b1 = 23.08336743743;
    static const double b2 = -21.06224101826;
    static const double b3 =  3.13082909833;

    // Moro tail coefficients (polynomial in r = log(-log(t)))
    static const double c0 = 0.3374754822726147;
    static const double c1 = 0.9761690190917186;
    static const double c2 = 0.1607979714918209;
    static const double c3 = 0.0276438810333863;
    static const double c4 = 0.0038405729373609;
    static const double c5 = 0.0003951896511919;
    static const double c6 = 0.0000321767881768;
    static const double c7 = 0.0000002888167364;
    static const double c8 = 0.0000003960315187;

    const double y = u - 0.5;
    if (std::fabs(y) < 0.42) {
        // Central rational approximation
        const double r = y * y;
        const double num = (((a3 * r + a2) * r + a1) * r + a0);
        const double den = ((((b3 * r + b2) * r + b1) * r + b0) * r + 1.0);
        return y * (num / den);
    } else {
        // Tails
        const double t = (y > 0.0) ? (1.0 - u) : u;   // in (0, 0.5]
        const double r = std::log(-std::log(t));
        double x = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*(c6 + r*(c7 + r*c8)))))));
        return (y < 0.0) ? -x : x;
    }
}

// Convenience: generate N standard normals from LCG + BSM
static inline vector<double> normals_from_LCG(size_t N, LCG& gen) {
    vector<double> z(N);
    for (size_t i = 0; i < N; ++i) {
        const double u = gen.next_uniform();   // (0,1)
        z[i] = inv_norm_cdf_BSM(u);           // N(0,1)
    }
    return z;
}
