#pragma once
#include <iostream>
#include <vector>
#include <tuple>
#include "../Linear_Algebra_operators/matrix.hpp"

class Inverse{

public:

    Inverse(Matrix A);

    Matrix test();
    Matrix compute();

private:
    Matrix A_;
};


// v.push_back(10);   // ajoute un élément à la fin
// v.pop_back();      // enlève le dernier élément
// v.size();          // nombre d’éléments
// v.empty();         // vrai si vide
// v.clear();         // vide le vecteur
// v[i];              // accès direct (rapide, pas de check)
// v.at(i);           // accès avec vérification (throw si hors bornes)
