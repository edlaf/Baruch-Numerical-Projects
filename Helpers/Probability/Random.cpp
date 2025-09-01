#include "Random.hpp"

namespace rng {
double uniform(std::uint64_t& state,
               std::uint64_t a, std::uint64_t c, std::uint64_t m) {
    state = (a * state + c) % m;
    return static_cast<double>(state) / static_cast<double>(m);
}

std::uint64_t uniform_(std::uint64_t& state,
                       std::uint64_t a, std::uint64_t c, std::uint64_t m) {
    state = (a * state + c) % m;
    return state;
}

Vector uniform(std::size_t N, std::uint64_t& state,
               std::uint64_t a, std::uint64_t c, std::uint64_t m) {
    std::vector<double> v; v.reserve(N);
    for (std::size_t i = 0; i < N; ++i) {
        auto s = uniform_(state, a, c, m);
        v.push_back(static_cast<double>(s) / static_cast<double>(m));
    }
    return Vector(v);
}
} // namespace rng
