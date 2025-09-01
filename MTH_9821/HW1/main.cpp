#include <iostream>
#include "../../Helpers/Linear_Algebra_operators/vector.hpp"
#include "../../Helpers/Linear_Algebra_operators/matrix.hpp"
#include "../../Helpers/Linear_Algebra_operators/operators.hpp"
#include "../../Helpers/Probability/Random.hpp"

int main() {
    std::cout << "--- HW1 ---" << std::endl;
    std::cout << "" << std::endl;

    std::cout << "- Question 1" << std::endl;
    std::size_t N = 10;
    std::uint64_t seed = 1;
    Vector sample = rng::uniform(N, seed);

    sample.print();

    return 0;
}
