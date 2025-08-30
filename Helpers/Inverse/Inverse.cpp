#include "Inverse.hpp"

Inverse::Inverse(Matrix A):A_(A){
    
}
Matrix Inverse::compute(){return A_;}

Matrix Inverse::test(){
    return A_;
}