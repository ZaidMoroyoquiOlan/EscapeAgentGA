#pragma once

#include <cuda_runtime.h>
#include "gene.h"

__global__ void createRandomChromosome(Gene* arr, int N, int M, int seed);

__global__ void myKernel();  // CUDA kernel declaration

void generatePopulation(Gene* h_arr, int N, int M);