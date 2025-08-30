#include <iostream>
#include "../../Helpers/Cholesky/Cholesky.hpp"

int main() {
    std::cout << "I am happy code in Cpp!\n";
    double x = 3.141592654;
    Cholesky zeub = Cholesky(x);
    std::cout << zeub.test() << std::endl;
    return 0;
}

// To run it:

// open terminal at the root of the repository (got there using cd)
// Then cd Numerical_Algebra and cd HW1
// run make everytime you do and udpate to the code
// run ./main to run the main.cpp

// If you are not coding with makefile then god bless you

// We can create a dll if needed to see the results in python or use cython I guess



