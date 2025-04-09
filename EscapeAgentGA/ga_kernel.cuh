#pragma once

#include <cuda_runtime.h>
#include "gene.h"

__global__ void createRandomChromosome(Gene* arr, int N, int M, int seed);

__global__ void EvaluateChromosomeKernel(Gene* chromosomes, float* results, float* sourcePos, float* sourceVel, float heading, float* targetPos, int NChromosomes, int NActions);

void generatePopulation(Gene* h_arr, int N, int M);

void EvaluateChromosomes(Gene* h_Chromosomes, float* h_Results, float* h_sourcePos, float* h_sourceVel, float heading, float* h_targetPos, int NChromosomes, int NActions);