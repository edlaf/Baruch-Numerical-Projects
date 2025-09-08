// down_and_out_mc.cpp
// Monte Carlo pricing of a down-and-out call (discrete monitoring) + closed-form benchmark.
// Usage: ./down_and_out_mc [lcg|mt] [seed]
//
// Build: g++ -O3 -std=c++17 down_and_out_mc.cpp -o down_and_out_mc
//
// Notes:
// - Closed-form identity used (no rebate), valid when B < min(S0, K):
//   C_do(S0,K,B) = C(S0,K) - (B/S0)^(2a) * C(B^2/S0, K),  a = (r - q)/sigma^2 - 1/2
// - MC simulates GBM under Q with discrete barrier checks at m steps.
// - “Optimal” m_k chosen as ceil( N_k^(1/3) * T^(2/3) ), as in many barrier MC assignments.

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>

// ------------------------- Math helpers -------------------------
inline double norm_cdf(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double bs_call(double S, double K, double T, double r, double q, double sigma) {
    if (T <= 0.0 || sigma <= 0.0) {
        return std::max(S * std::exp(-q * T) - K * std::exp(-r * T), 0.0);
    }
    double vsqrtT = sigma * std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrtT;
    double d2 = d1 - vsqrtT;
    return S * std::exp(-q * T) * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

double closed_form_down_and_out_call(double S, double K, double B, double T,
                                     double r, double q, double sigma) {
    // Requires B < min(S, K). S0=39, K=39, B=36 satisfies this.
    double a = (r - q) / (sigma * sigma) - 0.5;
    double mirrorS = (B * B) / S;
    return bs_call(S, K, T, r, q, sigma)
         - std::pow(B / S, 2.0 * a) * bs_call(mirrorS, K, T, r, q, sigma);
}

// ------------------------- LCG + Moro inverse -------------------------
struct LCGMoro {
    // x_{n+1} = (a * x_n + c) mod m, m = 2^31 - 1, a = 39373, c = 0
    // u_n = x_n / m, z_n = Phi^{-1}(u_n) via Beasley–Springer–Moro
    std::uint64_t a = 39373;
    std::uint64_t c = 0;
    std::uint64_t m = (1ull << 31) - 1ull;
    std::uint64_t state;

    explicit LCGMoro(std::uint64_t seed = 1) {
        if (seed == 0 || seed >= m) seed = 1;
        state = seed;
    }

    inline double uniform() {
        state = (a * state + c) % m;
        return static_cast<double>(state) / static_cast<double>(m);
    }

    static double moro_inv_cdf(double u) {
        // Coefficients from Moro’s approximation
        static const double A[4] = {
            2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637
        };
        static const double B[4] = {
            -8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833
        };
        static const double C[9] = {
            0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
            0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
            0.0000321767881768, 0.0000002888167364, 0.0000003960315187
        };

        if (u <= 0.0) return -INFINITY;
        if (u >= 1.0) return INFINITY;

        double y = u - 0.5;
        if (std::fabs(y) <= 0.42) {
            double r = y * y;
            double num = (((A[3] * r + A[2]) * r + A[1]) * r + A[0]);
            double den = ((((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0);
            return y * (num / den);
        } else {
            double w = (u > 0.5) ? (1.0 - u) : u;
            double r = std::log(-std::log(w));
            double x = C[0];
            double rp = 1.0;
            for (int i = 1; i < 9; ++i) {
                rp *= r;
                x += C[i] * rp;
            }
            return (u < 0.5) ? -x : x;
        }
    }
};

// ------------------------- Monte Carlo (streaming) -------------------------
double mc_down_and_out_call(std::size_t n_paths, int m_steps,
                            double S0, double K, double B, double T,
                            double r, double q, double sigma,
                            const std::string& engine = "lcg",
                            std::uint64_t seed = 42) {
    if (n_paths == 0 || m_steps <= 0) return std::numeric_limits<double>::quiet_NaN();

    double dt    = T / static_cast<double>(m_steps);
    double drift = (r - q - 0.5 * sigma * sigma) * dt;
    double vol   = sigma * std::sqrt(dt);
    double disc  = std::exp(-r * T);

    // RNG setup
    LCGMoro lcg(seed);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    double sum_payoff = 0.0;

    for (std::size_t i = 0; i < n_paths; ++i) {
        double S = S0;
        bool knocked = false;

        for (int j = 0; j < m_steps; ++j) {
            double z;
            if (engine == "lcg") {
                double u = lcg.uniform();
                z = LCGMoro::moro_inv_cdf(u);
            } else {
                z = nd(gen);
            }
            S *= std::exp(drift + vol * z);
            if (S <= B) { // down-and-out barrier hit
                knocked = true;
                break; // payoff will be 0, can early exit
            }
        }

        if (!knocked) {
            sum_payoff += std::max(0.0, S - K);
        }
    }
    return disc * (sum_payoff / static_cast<double>(n_paths));
}

// ------------------------- Experiment & main -------------------------
int main(int argc, char** argv) {
    std::string engine = "lcg";
    std::uint64_t seed = 42;
    if (argc >= 2) engine = argv[1];
    if (argc >= 3) seed = static_cast<std::uint64_t>(std::stoull(argv[2]));

    // Problem parameters (from prompt/images)
    const double S0 = 39.0;
    const double K  = 39.0;
    const double B  = 36.0;
    const double T  = 0.75;   // years
    const double r  = 0.02;
    const double q  = 0.01;
    const double sigma = 0.25;

    const double Cdao = closed_form_down_and_out_call(S0, K, B, T, r, q, sigma);

    std::cout.setf(std::ios::fixed);
    std::cout << std::setprecision(6);
    std::cout << "Closed-form C_dao = " << Cdao << "\n\n";
    std::cout << "   N_k   , m=200 ,     n    ,  V_hat(n) , |C_dao - V_hat(n)| ,  m_k ,   n_k  , V_hat(n_k) , |C_dao - V_hat(n_k)|\n";

    for (int k = 0; k <= 9; ++k) {
        std::uint64_t Nk = 10000ull << k;

        // (i) fixed m=200
        int m_fixed = 200;
        std::uint64_t n_fixed = Nk / static_cast<std::uint64_t>(m_fixed);
        if (n_fixed == 0) n_fixed = 1;
        double Vn_fixed = mc_down_and_out_call(n_fixed, m_fixed, S0, K, B, T, r, q, sigma, engine, seed);
        double err_fixed = std::fabs(Cdao - Vn_fixed);

        // (ii) "optimal" m_k, n_k
        int m_k = static_cast<int>(std::ceil(std::pow(static_cast<double>(Nk), 1.0 / 3.0) *
                                             std::pow(T, 2.0 / 3.0)));
        if (m_k < 1) m_k = 1;
        std::uint64_t n_k = Nk / static_cast<std::uint64_t>(m_k);
        if (n_k == 0) n_k = 1;

        // small change in seed for the second column to avoid perfect correlation
        double Vn_opt  = mc_down_and_out_call(n_k, m_k, S0, K, B, T, r, q, sigma, engine, seed + 1);
        double err_opt = std::fabs(Cdao - Vn_opt);

        std::cout << std::setw(8) << Nk << ", "
                  << std::setw(5) << m_fixed << " , "
                  << std::setw(7) << n_fixed << " , "
                  << std::setw(9) << Vn_fixed << " , "
                  << std::setw(19) << err_fixed << " , "
                  << std::setw(4) << m_k << " , "
                  << std::setw(6) << n_k << " , "
                  << std::setw(10) << Vn_opt << " , "
                  << std::setw(22) << err_opt << "\n";
    }

    return 0;
}
