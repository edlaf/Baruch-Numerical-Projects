#include "../Linear_Algebra_operators/vector.hpp"
#include "../Linear_Algebra_operators/matrix.hpp"
#include "../Linear_Algebra_operators/operators.hpp"
#include "../Cholesky/Cholesky.hpp"
#include "../Linear_system_solvers/Linear_solver.hpp"
#include "../Linear_system_solvers/Ordinary_least_square.hpp"

int main() {
// =========================
    // Jeu de données simple
    // y = 2 + 3x
    // =========================
    Matrix X({
        {1, 1},
        {1, 2},
        {1, 3},
        {1, 4}
    });
    Vector y({5, 8, 11, 14});  // vrai modèle : 2 + 3x

    // =========================
    // Créer le modèle OLS
    // =========================
    OLS model = Linear_Regression_OLS(X, y);

    // =========================
    // Entraîner le modèle
    // =========================
    model.fit();

    // =========================
    // Afficher les coefficients
    // =========================
    std::cout << "Coefficients estimés : " << model.coefs() << std::endl;

    // =========================
    // Prédictions sur X existant
    // =========================
    Vector preds = model.predict(X);
    std::cout << "Prédictions : " << preds << std::endl;

    // =========================
    // Prédiction pour un nouveau x = 5
    // =========================
    Vector x_new({1, 5});  // 1 = biais, 5 = valeur de x
    double y_pred = model.predict(x_new);
    std::cout << "Prédiction pour x=5 : " << y_pred << std::endl;

    return 0;
}



    // Matrix A({{4, 2},{2, 3}});
    // std::cout << "Matrix A:" << std::endl;
    // A.print();

    // Matrix L = Cholesky_decompose(A);
    // std::cout << "\nMatrix L (Cholesky factor):" << std::endl;
    // L.print();

    // Matrix approx = L * L.transpose();
    // std::cout << "\nReconstructed A = L * L^T:" << std::endl;
    // approx.print();

    // double error = (A - approx).norm();
    // std::cout << "\nVerification error ||A - L*L^T|| = " << error << std::endl;
    // A = Matrix ({
    //     {2, 1, 1},
    //     {4, -6, 0},
    //     {-2, 7, 2}
    // });

    // std::cout << "Matrix A:" << std::endl;
    // A.print();

    // auto LU_no_pivot = LU_decompose_without_pivot(A);
    // Matrix L1 = LU_no_pivot[0];
    // Matrix U1 = LU_no_pivot[1];

    // std::cout << "\nLU decomposition WITHOUT pivoting:" << std::endl;
    // std::cout << "L:" << std::endl;
    // L1.print();
    // std::cout << "U:" << std::endl;
    // U1.print();

    // double error1 = (A - (L1 * U1)).norm();
    // std::cout << "Verification error ||A - L*U|| = " << error1 << std::endl;

    // auto LU_pivot = LU_decompose_with_pivot(A);
    // Matrix P = LU_pivot[0];
    // Matrix L2 = LU_pivot[1];
    // Matrix U2 = LU_pivot[2];

    // std::cout << "\nLU decomposition WITH pivoting:" << std::endl;
    // std::cout << "P:" << std::endl;
    // P.print();
    // std::cout << "L:" << std::endl;
    // L2.print();
    // std::cout << "U:" << std::endl;
    // U2.print();

    // double error2 = (P*A - (L2 * U2)).norm();
    // std::cout << "Verification error ||PA - L*U|| = " << error2 << std::endl;

    // return 0;
