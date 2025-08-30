# Makefile
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I../../Helpers

TARGET := main
SRCS := main.cpp ../../Helpers/Cholesky/Cholesky.cpp

.PHONY: all run clean
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET)
