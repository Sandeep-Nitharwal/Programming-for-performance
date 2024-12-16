CXX=g++
CPPFLAGS=
CXXFLAGS=-O3 -std=c++20 -fopenmp -ltbb

# First target is the default target
all: problem1 problem2

# $@ is a variable that contains the target name.
problem1: problem1.cpp problem1_header.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

problem2: problem2.cpp problem2_header.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $(patsubst %.cpp,%.out,$<)

# The target is not a filename
clean:
	rm *.out compile_commands.json
