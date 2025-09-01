#include "matrix.hpp"
#include <algorithm>

Matrix::Matrix(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows, std::vector<double>(cols, 0.0)) {}
Matrix::Matrix(size_t rows, size_t cols, const std::vector<std::vector<double>>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data.size() != rows || (rows > 0 && data[0].size() != cols))
        throw std::runtime_error("Matrix size mismatch");
}
Matrix::Matrix(std::initializer_list<std::initializer_list<double>> values) {
    rows_ = values.size();
    cols_ = values.begin()->size();

    for (const auto& row : values) {
        if (row.size() != cols_) {
            throw std::runtime_error("Inconsistent row sizes in initializer list");
        }
    }

    data_.resize(rows_, std::vector<double>(cols_, 0.0));
    size_t i = 0;
    for (const auto& row : values) {
        size_t j = 0;
        for (const auto& val : row) {
            data_[i][j++] = val;
        }
        i++;
    }
}
Matrix::Matrix(const std::vector<Vector>& vectors) {
    if (vectors.empty()) throw std::runtime_error("No vectors provided");

    size_t n = vectors[0].size();
    size_t m = vectors.size();

    for (const auto& v : vectors) {
        if (v.size() != n) throw std::runtime_error("All vectors must have the same size");
    }

    rows_ = n;
    cols_ = m;
    data_ = std::vector<std::vector<double>>(rows_, std::vector<double>(cols_, 0.0));

    for (size_t j = 0; j < m; j++) {
        for (size_t i = 0; i < n; i++) {
            data_[i][j] = vectors[j][i];
        }
    }
}

size_t Matrix::rows() const { return rows_; }
size_t Matrix::cols() const { return cols_; }
double& Matrix::operator()(size_t i, size_t j) { return data_[i][j]; }
const double& Matrix::operator()(size_t i, size_t j) const { return data_[i][j]; }

void Matrix::print() const {
    std::cout << "[";
    for (size_t i=0;i<rows_;i++) {
        std::cout << "[";
        for (size_t j=0;j<cols_;j++) {
            std::cout << data_[i][j];
            if (j < cols_-1) std::cout << ", ";
        }
        std::cout << "]";
        if (i < rows_-1) std::cout << "\n ";
    }
    std::cout << "]" << std::endl;
}

Matrix Matrix::zeros(size_t rows, size_t cols) { return Matrix(rows, cols); }
Matrix Matrix::ones(size_t rows, size_t cols) { return Matrix(rows, cols, std::vector<std::vector<double>>(rows, std::vector<double>(cols, 1.0))); }
Matrix Matrix::random(size_t rows, size_t cols, double low, double high) {
    std::vector<std::vector<double>> data(rows, std::vector<double>(cols));
    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(low, high);
    for (size_t i=0;i<rows;i++) for (size_t j=0;j<cols;j++) data[i][j]=dis(gen);
    return Matrix(rows, cols, data);
}
Matrix Matrix::identity(size_t n) {
    Matrix I(n,n);
    for (size_t i=0;i<n;i++) I(i,i)=1.0;
    return I;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_!=other.rows_||cols_!=other.cols_) throw std::runtime_error("Matrix size mismatch");
    Matrix res(rows_, cols_);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) res(i,j)=data_[i][j]+other(i,j);
    return res;
}
Matrix Matrix::operator-(const Matrix& other) const {
    if (rows_!=other.rows_||cols_!=other.cols_) throw std::runtime_error("Matrix size mismatch");
    Matrix res(rows_, cols_);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) res(i,j)=data_[i][j]-other(i,j);
    return res;
}
Matrix Matrix::operator/(const Matrix& other) const {
    if (rows_!=other.rows_||cols_!=other.cols_) throw std::runtime_error("Matrix size mismatch");
    Matrix res(rows_, cols_);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) {
        if (std::abs(other(i,j))<1e-15) throw std::runtime_error("Division by zero");
        res(i,j)=data_[i][j]/other(i,j);
    }
    return res;
}

Matrix Matrix::operator+(double val) const {
    Matrix res(*this);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) res(i,j)+=val;
    return res;
}
Matrix Matrix::operator-(double val) const {
    Matrix res(*this);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) res(i,j)-=val;
    return res;
}
Matrix Matrix::operator*(double val) const {
    Matrix res(*this);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) res(i,j)*=val;
    return res;
}
Matrix Matrix::operator/(double val) const {
    if (std::abs(val)<1e-15) throw std::runtime_error("Division by zero");
    Matrix res(*this);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) res(i,j)/=val;
    return res;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) throw std::runtime_error("Matrix multiplication size mismatch");
    Matrix res(rows_, other.cols_);
    for (size_t i=0; i<rows_; i++) {
        for (size_t j=0; j<other.cols_; j++) {
            for (size_t k=0; k<cols_; k++) {
                res(i,j) += data_[i][k] * other(k,j);
            }
        }
    }
    return res;
}

Vector Matrix::operator*(const Vector& v) const {
    if (cols_ != v.size()) throw std::runtime_error("Matrix-Vector size mismatch");
    Vector res(std::vector<double>(rows_,0.0));
    for (size_t i=0;i<rows_;i++)
        for (size_t j=0;j<cols_;j++)
            res[i]+=data_[i][j]*v[j];
    return res;
}

Matrix operator*(const Vector& v, const Matrix& m) {
    if (v.size()!=m.rows()) throw std::runtime_error("Vector-Matrix size mismatch");
    Matrix res(1, m.cols());
    for (size_t j=0;j<m.cols();j++)
        for (size_t i=0;i<m.rows();i++)
            res(0,j)+=v[i]*m(i,j);
    return res;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) 
        throw std::runtime_error("Matrix size mismatch for Hadamard product");
    Matrix res(rows_, cols_);
    for (size_t i=0;i<rows_;i++) {
        for (size_t j=0;j<cols_;j++) {
            res(i,j) = data_[i][j] * other(i,j);
        }
    }
    return res;
}

Matrix Matrix::transpose() const {
    Matrix res(cols_, rows_);
    for (size_t i=0;i<rows_;i++) for (size_t j=0;j<cols_;j++) res(j,i)=data_[i][j];
    return res;
}

double Matrix::norm() const {
    double s = 0.0;
    for (size_t i = 0; i < rows_; i++) {
        for (size_t j = 0; j < cols_; j++) {
            s += data_[i][j] * data_[i][j];
        }
    }
    return std::sqrt(s);
}

void Matrix::swapRows(size_t i, size_t j) {
    if (i >= rows_ || j >= rows_)
        throw std::out_of_range("Row index out of range");
    std::swap(data_[i], data_[j]);
}

void Matrix::swapCols(size_t i, size_t j) {
    if (i >= cols_ || j >= cols_)
        throw std::out_of_range("Column index out of range");
    for (size_t r = 0; r < rows_; r++) {
        std::swap(data_[r][i], data_[r][j]);
    }
}

void Matrix::swapElements(size_t i1, size_t j1, size_t i2, size_t j2) {
    if (i1 >= rows_ || i2 >= rows_ || j1 >= cols_ || j2 >= cols_) {
        throw std::out_of_range("Matrix::swapElements indices out of range");
    }
    std::swap(data_[i1][j1], data_[i2][j2]);
}

Matrix Matrix::eye(size_t n) {
    return Matrix::identity(n);
}

bool Matrix::is_triangular(const Matrix& M, bool up) {
    if (M.rows() != M.cols()) {
        return false;
    }

    if (up) {
        for (size_t i = 0; i < M.rows(); i++) {
            for (size_t j = 0; j < i; j++) {
                if (std::abs(M(i, j)) > 1e-13) {
                    return false;
                }
            }
        }
        return true;
    } else {
        for (size_t i = 0; i < M.rows(); i++) {
            for (size_t j = i + 1; j < M.cols(); j++) {
                if (std::abs(M(i, j)) > 1e-13) {
                    return false;
                }
            }
        }
        return true;
    }
}

Matrix Matrix::add_intercept_column() const {
    Matrix Xnew(rows_, cols_ + 1);

    for (size_t i = 0; i < rows_; ++i) {
        Xnew(i, 0) = 1.0;
        for (size_t j = 0; j < cols_; ++j) {
            Xnew(i, j + 1) = data_[i][j];
        }
    }

    return Xnew;
}

