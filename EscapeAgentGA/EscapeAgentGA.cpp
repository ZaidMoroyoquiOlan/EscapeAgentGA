// EscapeAgentGA.cpp : Defines the entry point for the application.
//

#include <iostream>
#include "EscapeAgentGA.h"
#include "gene.h"
#include "ga_kernel.cuh"

using namespace std;

// Solution is the output
ESCAPEAGENTGA_API void RunCUDAGA(Gene *solution) {
	int n = 1, m = 10;
	
	generatePopulation(solution, n, m);
}
