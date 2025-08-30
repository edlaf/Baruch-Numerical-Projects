# Makefile (dans HW1/Helpers/)
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -MMD -MP -I. -ICholesky

TARGET := testing
SRCS := testing.cpp Cholesky/Cholesky.cpp
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
