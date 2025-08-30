#include "LU_decomposition.hpp"

LU::LU(Matrix A):A_(A){
    
}
Matrix LU::decompose(){
    return A_;
}

Matrix LU::test(){
    return A_;
}