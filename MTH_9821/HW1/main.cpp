#include <iostream>
#include "../../Helpers/Linear_Algebra_operators/vector.hpp"
#include "../../Helpers/Linear_Algebra_operators/matrix.hpp"
#include "../../Helpers/Linear_Algebra_operators/operators.hpp"
#include "../../Helpers/Probability/Random.hpp"

#include "../Helpers/TianTian_func/hw1.cpp"

int main() {
    std::cout << "--- HW1 ---" << std::endl;
    std::cout << "" << std::endl;

    std::cout << "- Question 1" << std::endl;
    std::size_t N = 10;
    std::uint64_t seed = 1;
    Vector sample = rng::uniform(N, seed);

    sample.print();

    std::cout << std::endl << "- Question 3" << std::endl;
    std::cout << "- Comparison of Random Number Generators" << std::endl;
    std::cout << "- Monte Carlo Valuation of Plain Vanilla Options" << std::endl;
    ComparisonOfRandomNumberGenerators_MonteCarloValuationOfPlainVanillaOptions();

    return 0;
}
