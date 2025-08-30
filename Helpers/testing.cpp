#include <iostream>
#include "../Cholesky/Cholesky.hpp"

int main() {
    std::cout << "I am happy code in Cpp!\n";
    double x = 3.141592654;
    Cholesky zeub = Cholesky(x);
    std::cout << zeub.test() << std::endl;
    return 0;
}
