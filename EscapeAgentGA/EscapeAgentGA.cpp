// EscapeAgentGA.cpp : Defines the entry point for the application.
//

#include <iostream>
#include "EscapeAgentGA.h"
#include "gene.h"
#include "ga_kernel.cuh"

using namespace std;

// Solution is the output, m is the size of the chromosome
ESCAPEAGENTGA_API void RunCUDAGA(Gene *solution, float* sourcePos, float* sourceVel, float heading, float* targetPos, int m) {
	int n = 128; // Size of population
	Gene* population = (Gene*)malloc(n * m * sizeof(Gene));
	float* results = (float*)malloc(n * sizeof(float));
	
	generatePopulation(population, n, m);

	EvaluateChromosomes(population, results, sourcePos, sourceVel, heading, targetPos, n, m);

	int bestScoreIdx = 0;
	for (int i = 1; i < n; i++) {
		if (results[i] > results[bestScoreIdx]) {
			bestScoreIdx = i;
		}
	}

	//printf("Best score [%d]: %f\n", bestScoreIdx, results[bestScoreIdx]);
	for (int i = 0; i < m; i++){
		solution[i] = population[m*bestScoreIdx+i];
	}
	free(population);
	free(results);
}
