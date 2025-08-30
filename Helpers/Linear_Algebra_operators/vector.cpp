#include "vector.hpp"

Vector::Vector(int n) : vect_(n, 0.0) {}

Vector::Vector(int n, const std::vector<double>& vect) : vect_(vect) {
    if (vect.size() != static_cast<size_t>(n)) throw std::runtime_error("Size mismatch");
}

Vector::Vector(const std::vector<double>& vect) : vect_(vect) {}

double& Vector::operator[](int index) {
    if (index < 0 || index >= static_cast<int>(vect_.size())) throw std::out_of_range("Index out of range");
    return vect_[index];
}

const double& Vector::operator[](int index) const {
    if (index < 0 || index >= static_cast<int>(vect_.size())) throw std::out_of_range("Index out of range");
    return vect_[index];
}

size_t Vector::size() const { return vect_.size(); }
void Vector::print() const { std::cout << *this << std::endl; }

Vector Vector::zeros(size_t n) { return Vector(std::vector<double>(n,0.0)); }
Vector Vector::ones(size_t n) { return Vector(std::vector<double>(n,1.0)); }

Vector Vector::arange(double start, double stop, double step) {
    std::vector<double> v;
    for (double x = start; x < stop; x += step) v.push_back(x);
    return Vector(v);
}

Vector Vector::linspace(double start, double stop, size_t num) {
    std::vector<double> v(num);
    double step = (stop-start)/(num-1);
    for (size_t i=0;i<num;i++) v[i] = start + i*step;
    return Vector(v);
}

Vector Vector::random(size_t n, double low, double high) {
    std::vector<double> v(n);
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(low, high);
    for (auto& x:v) x = dis(gen);
    return Vector(v);
}

Vector Vector::sum(const Vector& other) const {
    if (size()!=other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i=0;i<size();i++) res[i]=vect_[i]+other[i];
    return res;
}

Vector Vector::diff(const Vector& other) const {
    if (size()!=other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i=0;i<size();i++) res[i]=vect_[i]-other[i];
    return res;
}

Vector Vector::multiply(const Vector& other) const {
    if (size()!=other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i=0;i<size();i++) res[i]=vect_[i]*other[i];
    return res;
}

Vector Vector::div(const Vector& other) const {
    if (size()!=other.size()) throw std::runtime_error("Size mismatch");
    Vector res(size());
    for (size_t i=0;i<size();i++) {
        if (std::abs(other[i])<1e-15) throw std::runtime_error("Division by zero");
        res[i]=vect_[i]/other[i];
    }
    return res;
}

double Vector::sum() const { return std::accumulate(vect_.begin(), vect_.end(), 0.0); }
double Vector::prod() const { return std::accumulate(vect_.begin(), vect_.end(), 1.0, std::multiplies<double>()); }
void Vector::cumsum() { for (size_t i=1;i<size();i++) vect_[i]+=vect_[i-1]; }
void Vector::cumprod() { for (size_t i=1;i<size();i++) vect_[i]*=vect_[i-1]; }

void Vector::diff(int d) {
    if (d<=0) return;
    for (int k=0;k<d;k++) {
        for (size_t i=size()-1;i>0;i--) vect_[i]=vect_[i]-vect_[i-1];
        vect_[0]=0.0;
    }
}
Vector Vector::pow(int k) const {
    Vector res(size());
    for (size_t i=0;i<size();i++) res[i]=std::pow(vect_[i],k);
    return res;
}

double Vector::norm(int k) const { return std::pow(this->pow(k).sum(),1.0/k); }
double Vector::mean() const { return sum()/size(); }

double Vector::var() const {
    double m=mean(), v=0;
    for (double x:vect_) v+=(x-m)*(x-m);
    return v/size();
}

double Vector::std() const { return std::sqrt(var()); }
double Vector::min() const { return *std::min_element(vect_.begin(), vect_.end()); }
double Vector::max() const { return *std::max_element(vect_.begin(), vect_.end()); }
size_t Vector::argmin() const { return std::distance(vect_.begin(), std::min_element(vect_.begin(), vect_.end())); }
size_t Vector::argmax() const { return std::distance(vect_.begin(), std::max_element(vect_.begin(), vect_.end())); }

void Vector::add(double val){ vect_.push_back(val); }
void Vector::append(double val){ add(val); }
void Vector::remove(int index){ if(index<0||index>=static_cast<int>(vect_.size())) throw std::out_of_range("Index"); vect_.erase(vect_.begin()+index);}
void Vector::pop(int index){ remove(index); }
void Vector::sort(){ std::sort(vect_.begin(), vect_.end()); }

std::vector<size_t> Vector::argsort() const {
    std::vector<size_t> idx(size()); for(size_t i=0;i<size();i++) idx[i]=i;
    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j){return vect_[i]<vect_[j];});
    return idx;
}
void Vector::shift(int k){
    if(vect_.empty()) return;
    std::vector<double> res(size(),0.0);
    for(size_t i=0;i<size();i++){
        int newIndex=static_cast<int>(i)+k;
        if(newIndex>=0&&newIndex<static_cast<int>(size())) res[newIndex]=vect_[i];
    }
    vect_=res;
}
Vector Vector::slice(int start,int end,int step) const{
    if(start<0) start=0; if(end>static_cast<int>(size())) end=size();
    Vector res(0); std::vector<double> tmp;
    for(int i=start;i<end;i+=step) tmp.push_back(vect_[i]);
    return Vector(tmp);
}
Vector Vector::mask(const Vector& mask) const{
    if(size()!=mask.size()) throw std::runtime_error("Size mismatch");
    std::vector<double> tmp;
    for(size_t i=0;i<size();i++) if(mask[i]!=0.0) tmp.push_back(vect_[i]);
    return Vector(tmp);
}
void Vector::concat(const Vector& other){ vect_.insert(vect_.end(), other.vect_.begin(), other.vect_.end()); }

bool Vector::any() const { for(double v:vect_) if(v!=0.0) return true; return false; }
bool Vector::all() const { for(double v:vect_) if(v==0.0) return false; return true; }

std::vector<double> Vector::toStdVector() const { return vect_; }

Vector Vector::operator==(const Vector& other) const{ if(size()!=other.size()) throw std::runtime_error("Size mismatch"); Vector res(size()); for(size_t i=0;i<size();i++) res[i]=(vect_[i]==other[i]); return res;}
Vector Vector::operator!=(const Vector& other) const{ if(size()!=other.size()) throw std::runtime_error("Size mismatch"); Vector res(size()); for(size_t i=0;i<size();i++) res[i]=(vect_[i]!=other[i]); return res;}
Vector Vector::operator<(const Vector& other) const{ if(size()!=other.size()) throw std::runtime_error("Size mismatch"); Vector res(size()); for(size_t i=0;i<size();i++) res[i]=(vect_[i]<other[i]); return res;}
Vector Vector::operator<=(const Vector& other) const{ if(size()!=other.size()) throw std::runtime_error("Size mismatch"); Vector res(size()); for(size_t i=0;i<size();i++) res[i]=(vect_[i]<=other[i]); return res;}
Vector Vector::operator>(const Vector& other) const{ if(size()!=other.size()) throw std::runtime_error("Size mismatch"); Vector res(size()); for(size_t i=0;i<size();i++) res[i]=(vect_[i]>other[i]); return res;}
Vector Vector::operator>=(const Vector& other) const{ if(size()!=other.size()) throw std::runtime_error("Size mismatch"); Vector res(size()); for(size_t i=0;i<size();i++) res[i]=(vect_[i]>=other[i]); return res;}

Vector Vector::operator+(const Vector& other) const{ return sum(other); }
Vector Vector::operator-(const Vector& other) const{ return diff(other); }
Vector Vector::operator*(const Vector& other) const{ return multiply(other); }
Vector Vector::operator/(const Vector& other) const{ return div(other); }

Vector Vector::operator+(double val) const{ Vector res(*this); for(auto&v:res.vect_) v+=val; return res; }
Vector Vector::operator-(double val) const{ Vector res(*this); for(auto&v:res.vect_) v-=val; return res; }
Vector Vector::operator*(double val) const{ Vector res(*this); for(auto&v:res.vect_) v*=val; return res; }
Vector Vector::operator/(double val) const{ if(std::abs(val)<1e-15) throw std::runtime_error("Division by zero"); Vector res(*this); for(auto&v:res.vect_) v/=val; return res; }

Vector operator+(double val,const Vector& v){ return v+val; }
Vector operator-(double val,const Vector& v){ Vector res(v.size()); for(size_t i=0;i<v.size();i++) res[i]=val-v[i]; return res; }
Vector operator*(double val,const Vector& v){ return v*val; }
Vector operator/(double val,const Vector& v){ Vector res(v.size()); for(size_t i=0;i<v.size();i++){ if(std::abs(v[i])<1e-15) throw std::runtime_error("Division by zero"); res[i]=val/v[i]; } return res; }

std::ostream& operator<<(std::ostream& os, const Vector& v){
    os << "[";
    for(size_t i=0;i<v.size();i++){ os << v[i]; if(i<v.size()-1) os << ", "; }
    os << "]";
    return os;
}
