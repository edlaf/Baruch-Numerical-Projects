#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <algorithm>
using namespace std;

// -------- Normal pdf/cdf --------
inline double pdf(double x){ return exp(-0.5*x*x)/sqrt(2.0*M_PI); }
inline double cdf(double x){ return 0.5*erfc(-x*M_SQRT1_2); }

// -------- Black–Scholes (European put, dividend yield q) --------
struct BS { double V, Delta, Gamma, Theta; };
BS black_scholes_put(double S0,double K,double r,double q,double sigma,double T){
    double sqrtT = sqrt(T);
    double d1 = (log(S0/K) + (r - q + 0.5*sigma*sigma)*T)/(sigma*sqrtT);
    double d2 = d1 - sigma*sqrtT;
    double disc_r = exp(-r*T), disc_q = exp(-q*T);
    double V = K*disc_r*cdf(-d2) - S0*disc_q*cdf(-d1);
    double Delta = disc_q*(cdf(d1) - 1.0);
    double Gamma = disc_q * pdf(d1) / (S0*sigma*sqrtT);
    double Theta = -(S0*disc_q*pdf(d1)*sigma)/(2.0*sqrtT)
                   + q*S0*disc_q*cdf(-d1) - r*K*disc_r*cdf(-d2);
    return {V,Delta,Gamma,Theta};
}

// -------- Tree outputs --------
struct TG { double V, Delta, Gamma, Theta; };
struct Lattice { vector<vector<double>> S,U; };

TG greeks_from_lattice(const Lattice& L,double dt){
    auto S=[&](int i,int j){return L.S[i][j];};
    auto U=[&](int i,int j){return L.U[i][j];};
    double Delta=(U(1,0)-U(1,1))/(S(1,0)-S(1,1));
    double num=(U(2,0)-U(2,1))/(S(2,0)-S(2,1))-(U(2,1)-U(2,2))/(S(2,1)-S(2,2));
    double den=(S(2,0)-S(2,2))/2.0;
    double Gamma=num/den;
    double Theta=(U(2,1)-U(0,0))/(2.0*dt);
    return {U(0,0),Delta,Gamma,Theta};
}

TG crr_put(double S0,double K,double r,double q,double sigma,double T,int N){
    double dt=T/N,u=exp(sigma*sqrt(dt)),d=1.0/u;
    double disc=exp(-r*dt),a=exp((r-q)*dt),p=(a-d)/(u-d);
    Lattice L; L.S.assign(N+1,{}); L.U.assign(N+1,{});
    L.S[N].resize(N+1); L.U[N].resize(N+1);
    for(int j=0;j<=N;++j){L.S[N][j]=S0*pow(u,j)*pow(d,N-j);L.U[N][j]=max(K-L.S[N][j],0.0);}
    for(int i=N-1;i>=0;--i){
        L.S[i].resize(i+1);L.U[i].resize(i+1);
        for(int j=0;j<=i;++j){
            L.S[i][j]=S0*pow(u,j)*pow(d,i-j);
            L.U[i][j]=disc*(p*L.U[i+1][j+1]+(1-p)*L.U[i+1][j]);
        }
    }
    return greeks_from_lattice(L,dt);
}

TG bbs_put(double S0,double K,double r,double q,double sigma,double T,int N){
    double dt=T/N,mu=(r-q-0.5*sigma*sigma)*dt;
    double u=exp(mu+sigma*sqrt(dt)),d=exp(mu-sigma*sqrt(dt)),p=0.5,disc=exp(-r*dt);
    Lattice L; L.S.assign(N+1,{}); L.U.assign(N+1,{});
    L.S[N].resize(N+1);L.U[N].resize(N+1);
    for(int j=0;j<=N;++j){L.S[N][j]=S0*pow(u,j)*pow(d,N-j);L.U[N][j]=max(K-L.S[N][j],0.0);}
    for(int i=N-1;i>=0;--i){
        L.S[i].resize(i+1);L.U[i].resize(i+1);
        for(int j=0;j<=i;++j){
            L.S[i][j]=S0*pow(u,j)*pow(d,i-j);
            L.U[i][j]=disc*(p*L.U[i+1][j+1]+(1-p)*L.U[i+1][j]);
        }
    }
    return greeks_from_lattice(L,dt);
}

TG avg_crr(double S0,double K,double r,double q,double sigma,double T,int N){
    TG a=crr_put(S0,K,r,q,sigma,T,N),b=crr_put(S0,K,r,q,sigma,T,N+1);
    return {(a.V+b.V)/2.0,(a.Delta+b.Delta)/2.0,(a.Gamma+b.Gamma)/2.0,(a.Theta+b.Theta)/2.0};
}

TG bbs_richardson(double S0,double K,double r,double q,double sigma,double T,int N){
    TG a=bbs_put(S0,K,r,q,sigma,T,N),b=bbs_put(S0,K,r,q,sigma,T,N/2);
    return {2*a.V-b.V,2*a.Delta-b.Delta,2*a.Gamma-b.Gamma,2*a.Theta-b.Theta};
}

int main(){
    double S0=100,K=100,r=0.05,q=0,sigma=0.2,T=1.0;
    ofstream fout("hw3_result_filtered.csv");
    fout.setf(ios::fixed); fout<<setprecision(10);
    BS bs=black_scholes_put(S0,K,r,q,sigma,T);

    // top
    fout<<"Binomial Tree Methods for European Options\n\n";
    fout<<"V_BS,,"<<bs.V<<",,,Delta_BS,,"<<bs.Delta
        <<",,,Gamma_BS,,"<<bs.Gamma<<",,,Theta_BS,,"<<bs.Theta<<"\n\n";

    auto head=[&](const string&t){fout<<t<<"\nN,V(N),|V(N)-V_BS|,N*|V(N)-V_BS|,N^2*|V(N)-V_BS|,"
                                       <<"Delta_1,|Delta_1-Delta_BS|,Gamma_1,|Gamma_1-Gamma_BS|,"
                                       <<"Theta_1,|Theta_1-Theta_BS|\n";};
    auto row=[&](int N,TG t){double e=fabs(t.V-bs.V);
        fout<<N<<","<<t.V<<","<<e<<","<<N*e<<","<<N*N*e<<","
            <<t.Delta<<","<<fabs(t.Delta-bs.Delta)<<","
            <<t.Gamma<<","<<fabs(t.Gamma-bs.Gamma)<<","
            <<t.Theta<<","<<fabs(t.Theta-bs.Theta)<<"\n";};

    vector<int> Ns={10,20,40,80,160,320,640,1280};

    head("Binomial Tree");
    for(int N:Ns) row(N,crr_put(S0,K,r,q,sigma,T,N));
    fout<<"\n";

    head("Average Binomial Tree");
    for(int N:Ns) row(N,avg_crr(S0,K,r,q,sigma,T,N));
    fout<<"\n";

    head("Binomial Black–Scholes");
    for(int N:Ns) row(N,bbs_put(S0,K,r,q,sigma,T,N));
    fout<<"\n";

    head("Binomial Black–Scholes with Richardson Extrapolation");
    for(int N:vector<int>{10,20,40,80,160,320,640,1280}) row(N,bbs_richardson(S0,K,r,q,sigma,T,N));

    fout.close();
    return 0;
}
