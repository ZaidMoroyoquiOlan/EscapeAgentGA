#pragma once

#include <cuda_runtime.h>
#include "gene.h"

void ExecuteGA(Gene* solution, float* sourcePos, float* sourceVel, float heading, float* targetPos, float targetHeading, int NChromosomes, int NActions, int NGenerations);