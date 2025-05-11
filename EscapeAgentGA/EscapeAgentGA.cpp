// EscapeAgentGA.cpp : Defines the entry point for the application.
//

#include <iostream>
#include "EscapeAgentGA.h"
#include "gene.h"
#include "ga_kernel.cuh"

using namespace std;

ESCAPEAGENTGA_API void InitCudaArray(const uint8_t* binArray, size_t size) {
	AllocateMap(binArray, size);
}

// Solution is the output, m is the size of the chromosome
ESCAPEAGENTGA_API void RunCUDAGA(Gene *solution, float* sourcePos, float* sourceVel, float heading, float* targetPos, float targetHeading, int m) {
	int n = 256; // Size of population
	int Generations = 100;

	ExecuteGA(solution, sourcePos, sourceVel, heading, targetPos, targetHeading, n, m, Generations);
}

ESCAPEAGENTGA_API void CleanupCuda() {
	FreeMap();
}