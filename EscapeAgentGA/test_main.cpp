#include <iostream>
#include "EscapeAgentGA.h"
#include "gene.h"

// This file is not important it is just used for testing
int main() {
    std::cout << "Running test executable." << std::endl;
    int sizeOfChromosome = 10;
    Gene* array = (Gene*)malloc(sizeOfChromosome * sizeof(Gene));

    float* sourcePos = (float*)malloc(2 * sizeof(float));
    sourcePos[0] = 0.0f;
    sourcePos[1] = 0.0f;

    float* sourceVel = (float*)malloc(2 * sizeof(float));
    sourceVel[0] = 0.0f;
    sourceVel[1] = 0.0f;

    float* targetPos = (float*)malloc(2 * sizeof(float));
    targetPos[0] = 0.0f;
    targetPos[1] = -1000.0f;

    RunCUDAGA(array, sourcePos, sourceVel, 0.0f, targetPos, 0.0f, sizeOfChromosome);

    for (int i = 0; i < sizeOfChromosome; i++) {
        std::cout << i << " " << array[i].throttle << " " << array[i].direction << std::endl;
    }

    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}
