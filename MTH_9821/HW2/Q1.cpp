#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

// ====== Toggle Excel writing ======
// If you install xlnt (`brew install xlnt`), compile with -DUSE_XLNT and link it.
// Example:
//   brew install xlnt
//   g++ -std=c++17 -O3 mc_vr_hw2.cpp -o mc_vr_hw2 -I/usr/local/include -L/usr/local/lib -lxlnt
// If not using xlnt, comment out the define below or compile without -DUSE_XLNT; CSVs will be written instead.
// #define USE_XLNT

#ifdef USE_XLNT
#include <xlnt/xlnt.hpp>
#endif

// ---------- Problem inputs ----------
struct Params {
    double S0 = 56.0;
    double K  = 54.0;
    double r  = 0.02;
    double T  = 0.75;     // 9 months
    double sig= 0.27;
};

// ---------- Normal CDF ----------
static inline double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

// ---------- Black–Scholes European put ----------
double bs_put(const Params& p) {
    double sqrtT = std::sqrt(p.T);
    double d1 = (std::log(p.S0 / p.K) + (p.r + 0.5 * p.sig * p.sig) * p.T) / (p.sig * sqrtT);
    double d2 = d1 - p.sig * sqrtT;
    // Put: K e^{-rT} N(-d2) - S0 N(-d1)
    return p.K * std::exp(-p.r * p.T) * norm_cdf(-d2) - p.S0 * norm_cdf(-d1);
}

// ---------- LCG (a=39373, c=0, k=2^31-1), u in (0,1) ----------
struct LCG {
    using u64 = uint64_t;
    static const u64 a = 39373ULL;
    static const u64 c = 0ULL;
    static const u64 m = 2147483647ULL; // 2^31 - 1 (prime)
    u64 x;

    explicit LCG(u64 seed=1ULL): x(seed % m) { if (x==0) x=1; }

    inline double next_u01() {
        // x_{i+1} = (a*x_i + c) mod m
        x = (a * x) % m;
        // Map to (0,1); avoid exact 0
        return (double)x / (double)m;
    }
};

// ---------- Marsaglia–Bray polar method (Box–Muller) ----------
// Produces pairs of independent N(0,1)
struct PolarNormal {
    LCG& rng;
    bool has_spare = false;
    double spare   = 0.0;

    explicit PolarNormal(LCG& lcg) : rng(lcg) {}

    inline double next() {
        if (has_spare) { has_spare = false; return spare; }
        double u, v, s;
        do {
            double u1 = rng.next_u01();
            double u2 = rng.next_u01();
            u = 2.0*u1 - 1.0;
            v = 2.0*u2 - 1.0;
            s = u*u + v*v;
        } while (s <= 0.0 || s >= 1.0);
        double factor = std::sqrt(-2.0 * std::log(s) / s);
        spare = v * factor;
        has_spare = true;
        return u * factor;
    }
};

// ---------- Payoff helpers ----------
static inline double ST_from_z(const Params& p, double z) {
    double mu = (p.r - 0.5 * p.sig * p.sig) * p.T;
    double sd = p.sig * std::sqrt(p.T);
    return p.S0 * std::exp(mu + sd * z);
}
static inline double disc_put_payoff(const Params& p, double ST) {
    return std::exp(-p.r * p.T) * std::max(p.K - ST, 0.0);
}

// ---------- One run (for a given n), producing all 4 estimators ----------
struct Estimators {
    double V_CV      = 0.0;
    double V_AV      = 0.0;
    double V_MM      = 0.0;
    double V_CVMM    = 0.0;
};

Estimators run_all_estimators(std::size_t n, const Params& par, PolarNormal& N01) {
    // For CV and MM, we need full vectors (for b-hat and mean)
    std::vector<double> S(n);
    std::vector<double> V(n);

    // --- Generate normals and base payoffs ---
    for (std::size_t i=0; i<n; ++i) {
        double z = N01.next();
        S[i] = ST_from_z(par, z);
        V[i] = disc_put_payoff(par, S[i]);
    }

    // Precompute constants
    const double Se = std::exp(par.r * par.T) * par.S0; // E[S_T] = S0 e^{rT}
    // Sample means
    double meanS = 0.0, meanV = 0.0;
    for (std::size_t i=0; i<n; ++i) { meanS += S[i]; meanV += V[i]; }
    meanS /= double(n);
    meanV /= double(n);

    // --- Control Variate (CV) ---
    // b_hat = Cov(S,V) / Var(S)
    double num=0.0, den=0.0;
    for (std::size_t i=0; i<n; ++i) {
        double ds = S[i] - meanS;
        double dv = V[i] - meanV;
        num += ds * dv;
        den += ds * ds;
    }
    double b_hat = (den > 0.0 ? num / den : 0.0);

    double W_sum = 0.0;
    for (std::size_t i=0; i<n; ++i) {
        double Wi = V[i] - b_hat * (S[i] - Se);
        W_sum += Wi;
    }
    double V_CV = W_sum / double(n);

    // --- Antithetic Variates (AV) ---
    // regenerate with pairs z and -z
    // We’ll produce n samples as n pairs averaged; if n is odd, drop one.
    std::size_t m = n / 2;
    double sumAV = 0.0;
    for (std::size_t j=0; j<m; ++j) {
        double z1 = N01.next();
        double z2 = -z1;
        double S1 = ST_from_z(par, z1);
        double S2 = ST_from_z(par, z2);
        double V1 = disc_put_payoff(par, S1);
        double V2 = disc_put_payoff(par, S2);
        sumAV += 0.5 * (V1 + V2);
    }
    double V_AV = (m>0 ? sumAV / double(m) : 0.0);

    // --- Moment Matching (MM) ---
    // Scale S by factor to match mean Se, then recompute V~ on the scaled S
    double scale = (meanS != 0.0 ? Se / meanS : 1.0);
    double sumMM = 0.0;
    for (std::size_t i=0; i<n; ++i) {
        double S_tilde = S[i] * scale;
        double V_tilde = disc_put_payoff(par, S_tilde);
        sumMM += V_tilde;
    }
    double V_MM = sumMM / double(n);

    // --- Simultaneous MM + CV ---
    // Redo CV on the tilde variables
    double meanSt = Se;                // by construction after scaling
    double meanVt = V_MM;
    double num2=0.0, den2=0.0;
    for (std::size_t i=0; i<n; ++i) {
        double S_tilde = S[i] * scale;
        double V_tilde = disc_put_payoff(par, S_tilde);
        double ds = (S_tilde - meanSt);
        double dv = (V_tilde - meanVt);
        num2 += ds * dv;
        den2 += ds * ds;
    }
    double b2 = (den2 > 0.0 ? num2 / den2 : 0.0);
    double sumCVMM = 0.0;
    for (std::size_t i=0; i<n; ++i) {
        double S_tilde = S[i] * scale;
        double V_tilde = disc_put_payoff(par, S_tilde);
        double Wi = V_tilde - b2 * (S_tilde - Se);
        sumCVMM += Wi;
    }
    double V_CVMM = sumCVMM / double(n);

    return { V_CV, V_AV, V_MM, V_CVMM };
}

// ---------- Excel / CSV writing ----------
struct RowAV { std::size_t n; double Vhat; double err; };
struct RowCombo { std::size_t n; double V_CVMM, err_CVMM, V_CV, err_CV, V_MM, err_MM; };

#ifdef USE_XLNT
void write_to_excel(const std::string& xlsx_path,
                    const std::vector<RowAV>& av_rows,
                    const std::vector<RowCombo>& combo_rows)
{
    xlnt::workbook wb;
    wb.load(xlsx_path);
    auto ws = wb.sheet_by_title("Problem 1");

    // Antithetic table header found at row 6: (1) n, (2) V^_AV(N), (3) |V_BS – V^_AV(N)|
    // Data rows are 7..16 for n = 10k..5.12m
    int av_start_row = 7;
    for (std::size_t i=0; i<av_rows.size(); ++i) {
        int r = av_start_row + static_cast<int>(i);
        ws.cell("A" + std::to_string(r)).value(static_cast<double>(av_rows[i].n));
        ws.cell("B" + std::to_string(r)).value(av_rows[i].Vhat);
        ws.cell("C" + std::to_string(r)).value(av_rows[i].err);
    }

    // Combined table header at row 20/21: columns:
    // A: n, B: V^_CV,MM(N), C: |...|, D: V^_CV(N), E: |...|, F: V^_MM(N), G: |...|
    int combo_start_row = 21;
    for (std::size_t i=0; i<combo_rows.size(); ++i) {
        int r = combo_start_row + static_cast<int>(i);
        ws.cell("A" + std::to_string(r)).value(static_cast<double>(combo_rows[i].n));
        ws.cell("B" + std::to_string(r)).value(combo_rows[i].V_CVMM);
        ws.cell("C" + std::to_string(r)).value(combo_rows[i].err_CVMM);
        ws.cell("D" + std::to_string(r)).value(combo_rows[i].V_CV);
        ws.cell("E" + std::to_string(r)).value(combo_rows[i].err_CV);
        ws.cell("F" + std::to_string(r)).value(combo_rows[i].V_MM);
        ws.cell("G" + std::to_string(r)).value(combo_rows[i].err_MM);
    }

    wb.save(xlsx_path); // overwrite in place
}
#else
void write_csvs(const std::vector<RowAV>& av_rows,
                const std::vector<RowCombo>& combo_rows)
{
    {
        std::ofstream out("Problem1_Antithetic.csv");
        out << "n,Vhat_AV,abs_err\n";
        for (auto& r : av_rows) {
            out << r.n << "," << r.Vhat << "," << r.err << "\n";
        }
    }
    {
        std::ofstream out("Problem1_CV_MM_and_CVMM.csv");
        out << "n,Vhat_CVMM,abs_err_CVMM,Vhat_CV,abs_err_CV,Vhat_MM,abs_err_MM\n";
        for (auto& r : combo_rows) {
            out << r.n << "," << r.V_CVMM << "," << r.err_CVMM << ","
                << r.V_CV << "," << r.err_CV << ","
                << r.V_MM << "," << r.err_MM << "\n";
        }
    }
    std::cout << "[INFO] Wrote CSVs: Problem1_Antithetic.csv, Problem1_CV_MM_and_CVMM.csv\n";
}
#endif

int main(int argc, char** argv) {
    // Paths
    std::string xlsx_path = "hw2_9821_fall2025_mc2-blueprint.xlsx"; // put the Excel in the same folder

    Params p;
    double VBS = bs_put(p);
    std::cout << "Black–Scholes put price (benchmark): " << VBS << "\n";

    // Random generator (seeded deterministically for reproducibility)
    uint64_t seed = 1ULL;
    if (argc >= 2) {
        // optional custom seed
        seed = std::stoull(argv[1]);
    }
    LCG lcg(seed);
    PolarNormal N01(lcg);

    // n-list: 10k .. 5.12m, doubling
    std::vector<std::size_t> N_list;
    std::size_t n = 10000;
    for (int i=0; i<10; ++i) { // 10 rows: 10k .. 5,120,000
        N_list.push_back(n);
        n *= 2;
    }

    std::vector<RowAV> av_rows;
    std::vector<RowCombo> combo_rows;

    for (auto N : N_list) {
        auto est = run_all_estimators(N, p, N01);
        // errors
        double eCV   = std::fabs(VBS - est.V_CV);
        double eAV   = std::fabs(VBS - est.V_AV);
        double eMM   = std::fabs(VBS - est.V_MM);
        double eCVMM = std::fabs(VBS - est.V_CVMM);

        av_rows.push_back({ N, est.V_AV, eAV });
        combo_rows.push_back({ N, est.V_CVMM, eCVMM, est.V_CV, eCV, est.V_MM, eMM });

        std::cout << "n=" << N
                  << "  CV="   << est.V_CV   << " |err|=" << eCV
                  << "  AV="   << est.V_AV   << " |err|=" << eAV
                  << "  MM="   << est.V_MM   << " |err|=" << eMM
                  << "  CVMM=" << est.V_CVMM << " |err|=" << eCVMM
                  << "\n";
    }

#ifdef USE_XLNT
    try {
        write_to_excel(xlsx_path, av_rows, combo_rows);
        std::cout << "[INFO] Wrote results into Excel template: " << xlsx_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Excel write failed (" << e.what() << "). Writing CSVs instead.\n";
        write_csvs(av_rows, combo_rows);
    }
#else
    write_csvs(av_rows, combo_rows);
#endif

    return 0;
}
