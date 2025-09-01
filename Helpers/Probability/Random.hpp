#ifndef PROBABILITY_RANDOM_HPP
#define PROBABILITY_RANDOM_HPP

#include <cstdint>
#include <cstddef>
#include <vector>
#include "../Linear_Algebra_operators/vector.hpp"

namespace rng {

/**
 * @brief Generate a uniform variable with a Linear Congruential Generator (LCG)
 * @return Uniform variable in [0,1)
 */
double uniform(std::uint64_t& state,
               std::uint64_t a = 39373,
               std::uint64_t c = 0,
               std::uint64_t m = 2147483647ULL);

std::uint64_t uniform_(std::uint64_t& state,
                       std::uint64_t a = 39373,
                       std::uint64_t c = 0,
                       std::uint64_t m = 2147483647ULL);

/**
 * @brief Generate N uniform variables with the same LCG
 * @param N Number of variables
 * @param state Seed/state (updated in-place)
 * @return A Vector containing N uniforms in [0,1)
 */
Vector uniform(std::size_t N, std::uint64_t& state,
               std::uint64_t a = 39373,
               std::uint64_t c = 0,
               std::uint64_t m = 2147483647ULL);

}

#endif
