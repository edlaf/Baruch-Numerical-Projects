CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -MMD -MP -I. -ICholesky

TARGET := testing
SRCS := testing.cpp \
        Cholesky/Cholesky.cpp \
		LU_decomposition/LU_decomposition.cpp \
        Linear_Algebra_operators/vector.cpp \
        Linear_Algebra_operators/matrix.cpp \
		Linear_Algebra_operators/operators.cpp \
		Linear_system_solvers/Linear_solver.cpp \
		Linear_system_solvers/Ordinary_least_square.cpp \
		Inverse/Inverse.cpp \
		Regressor/Cubic_splines/cubic_splines.cpp \

OBJS := $(SRCS:.cpp=.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all run clean
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@

-include $(DEPS)

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET) $(OBJS) $(DEPS)
