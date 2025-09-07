#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <limits>
#include <algorithm>

using namespace std;

// ======================= Utilities & Models =======================

// -------- Linear Congruential Generator --------
struct LCG {
    using u64 = uint64_t;
    static constexpr u64 A = 39373ULL;              // multiplier
    static constexpr u64 C = 0ULL;                  // increment
    static constexpr u64 K = (1ULL << 31) - 1ULL;   // modulus
    u64 x;
    explicit LCG(u64 seed=1ULL): x(seed % K) {}
    inline double next_uniform() {
        x = (A * x + C) % K;
        return ((double)x + 0.5) / (double)K;       // uniform in (0,1)
    }
};

// -------- Inverse Normal CDF (Moro) --------
static inline double inv_norm_cdf_moro(double u) {
    static const double a[4] = {2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637};
    static const double b[4] = {-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833};
    static const double c[9] = {
        0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
        0.0276438810333863, 0.0038405729373609, 0.0003951896511919,
        0.0000321767881768, 0.0000002888167364, 0.0000003960315187
    };
    double x = u - 0.5;
    if (fabs(x) <= 0.42) {
        double r = x * x;
        double num = (((a[3]*r + a[2]) * r + a[1]) * r + a[0]) * x;
        double den = ((((b[3]*r + b[2]) * r + b[1]) * r + b[0]) * r + 1.0);
        return num / den;
    } else {
        double y = (x > 0.0) ? 1.0 - u : u;
        double z = log(-log(y));
        double ret = c[0] + z*(c[1] + z*(c[2] + z*(c[3] + z*(c[4] + z*(c[5] + z*(c[6] + z*(c[7] + z*c[8])))))));
        return (x < 0.0) ? -ret : ret;
    }
}

// -------- Normal CDF --------
static inline double norm_cdf(double x) {
    return 0.5 * erfc(-x / M_SQRT2);
}

// -------- Black–Scholes Put --------
double bs_put(double S0, double K, double r, double q, double sigma, double T) {
    double sT = sigma * sqrt(T);
    double d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / sT;
    double d2 = d1 - sT;
    return K * exp(-r * T) * norm_cdf(-d2) - S0 * exp(-q * T) * norm_cdf(-d1);
}

// -------- Normal Generators --------
vector<double> gen_norm_inverse_transform(size_t N, LCG &rng) {
    vector<double> z; z.reserve(N);
    for (size_t i = 0; i < N; ++i) z.push_back(inv_norm_cdf_moro(rng.next_uniform()));
    return z;
}

vector<double> gen_norm_accept_reject_budgeted(size_t N_uniform_budget, LCG &rng) {
    vector<double> z; z.reserve(N_uniform_budget / 3);
    size_t used = 0;
    while (used + 3 <= N_uniform_budget) {
        double u1 = rng.next_uniform(); used++;
        double u2 = rng.next_uniform(); used++;
        double u3 = rng.next_uniform(); used++;
        double X = -log(u1);
        if (u2 > exp(-0.5 * (X - 1.0) * (X - 1.0))) continue;
        if (u3 <= 0.5) X = -X;
        z.push_back(X);
    }
    return z;
}

vector<double> gen_norm_box_muller_budgeted(size_t N_uniform_budget, LCG &rng) {
    vector<double> z; z.reserve(N_uniform_budget / 2);
    size_t used = 0;
    while (used + 2 <= N_uniform_budget) {
        double u1 = rng.next_uniform(); used++;
        double u2 = rng.next_uniform(); used++;
        double x = 2.0 * u1 - 1.0;
        double y = 2.0 * u2 - 1.0;
        double s = x*x + y*y;
        if (s <= 0.0 || s >= 1.0) continue;
        double factor = sqrt(-2.0 * log(s) / s);
        z.push_back(x * factor);
        z.push_back(y * factor);
    }
    return z;
}

// -------- Monte Carlo Put Pricing --------
double mc_put_estimate(const vector<double>& zs,
                       double S0, double K, double r, double q, double sigma, double T)
{
    if (zs.empty()) return numeric_limits<double>::quiet_NaN();
    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double vol   = sigma * sqrt(T);
    double disc  = exp(-r * T);
    long double acc = 0.0L;
    for (double z : zs) {
        double ST = S0 * exp(drift + vol * z);
        double payoff = max(K - ST, 0.0);
        acc += payoff;
    }
    return (double)(disc * acc / (long double)zs.size());
}

// ======================= Main Comparison Function =======================
void ComparisonOfRandomNumberGenerators_MonteCarloValuationOfPlainVanillaOptions() {
    const double S0 = 50.0, K = 55.0, T = 0.5, sigma = 0.3, r = 0.04, q = 0.0;
    const double VBS = bs_put(S0, K, r, q, sigma, T);

    cout.setf(std::ios::fixed); cout << setprecision(6);
    cout << "Black–Scholes Put: V_BS = " << VBS << "\n\n";

    // Inverse Transform
    cout << "Inverse Transform Method\n";
    cout << "N, Vhat(N), |V_BS - Vhat(N)|\n";
    for (int k = 0; k <= 9; ++k) {
        size_t N = 10000ULL << k;
        LCG rng(1ULL);
        auto zs = gen_norm_inverse_transform(N, rng);
        double Vhat = mc_put_estimate(zs, S0, K, r, q, sigma, T);
        cout << N << ", " << Vhat << ", " << fabs(VBS - Vhat) << "\n";
    }
    cout << "\n";

    // Acceptance–Rejection
    cout << "Acceptance–Rejection Method (budgeted)\n";
    cout << "N, N_A-R, Vhat(N_A-R), |V_BS - Vhat(N_A-R)|\n";
    for (int k = 0; k <= 9; ++k) {
        size_t N_budget = 10000ULL << k;
        LCG rng(1ULL);
        auto zs = gen_norm_accept_reject_budgeted(N_budget, rng);
        double Vhat = mc_put_estimate(zs, S0, K, r, q, sigma, T);
        cout << N_budget << ", " << zs.size() << ", "
             << Vhat << ", " << fabs(VBS - Vhat) << "\n";
    }
    cout << "\n";

    // Box–Muller
    cout << "Box–Muller Method (budgeted)\n";
    cout << "N, N_B-M, Vhat(N_B-M), |V_BS - Vhat(N_B-M)|\n";
    for (int k = 0; k <= 9; ++k) {
        size_t N_budget = 10000ULL << k;
        LCG rng(1ULL);
        auto zs = gen_norm_box_muller_budgeted(N_budget, rng);
        double Vhat = mc_put_estimate(zs, S0, K, r, q, sigma, T);
        cout << N_budget << ", " << zs.size() << ", "
             << Vhat << ", " << fabs(VBS - Vhat) << "\n";
    }
    cout << "\n";
}
