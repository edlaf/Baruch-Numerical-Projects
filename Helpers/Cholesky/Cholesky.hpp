#pragma once
#include <iostream>
#include <vector>
#include <tuple>
class Cholesky{

public:

    Cholesky();
    Cholesky(double x);

    double test();

private:
    double x_;
};


// v.push_back(10);   // ajoute un élément à la fin
// v.pop_back();      // enlève le dernier élément
// v.size();          // nombre d’éléments
// v.empty();         // vrai si vide
// v.clear();         // vide le vecteur
// v[i];              // accès direct (rapide, pas de check)
// v.at(i);           // accès avec vérification (throw si hors bornes)
