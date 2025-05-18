#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include "EscapeAgentGA.h"
#include "gene.h"

// Este archivo solo se usa para pruebas
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

    const size_t binSize = 252*250;
    std::vector<uint8_t> binArray(binSize);
    for (size_t i = 0; i < binSize; ++i) {
        binArray[i] = static_cast<uint8_t>(1);
    }

    // Medir tiempo de ejecución
    auto start = std::chrono::high_resolution_clock::now();

    InitCudaArray(binArray.data(), binSize);
    RunCUDAGA(array, sourcePos, sourceVel, 0.0f, targetPos, 0.0f, sizeOfChromosome);
    CleanupCuda();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "RunCUDAGA tiempo de ejecución " << duration.count() << " ms." << std::endl;

    for (int i = 0; i < sizeOfChromosome; i++) {
        std::cout << i << " " << array[i].throttle << " " << array[i].direction << std::endl;
    }

    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}
