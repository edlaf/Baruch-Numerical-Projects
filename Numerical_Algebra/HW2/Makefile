# Makefile
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I../../Helpers

TARGET := main
SRCS := main.cpp \
        ../../Helpers/Cholesky/Cholesky.cpp \
		../../Helpers/LU_decomposition/LU_decomposition.cpp \
        ../../Helpers/Linear_Algebra_operators/vector.cpp \
        ../../Helpers/Linear_Algebra_operators/matrix.cpp \
		../../Helpers/Linear_Algebra_operators/operators.cpp \
		../../Helpers/Linear_system_solvers/Linear_solver.cpp \
		../../Helpers/Linear_system_solvers/Ordinary_least_square.cpp \
		../../Helpers/Inverse/Inverse.cpp \
		../../Helpers/Regressor/Cubic_splines/cubic_splines.cpp \

.PHONY: all run clean
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)
