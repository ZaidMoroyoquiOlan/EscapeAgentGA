#include <iostream>
#include "EscapeAgentGA.h"
#include "gene.h"

int main() {
    std::cout << "Running test executable." << std::endl;
    Gene* array = (Gene*)malloc(1 * 10 * sizeof(Gene));

    RunCUDAGA(array);

    for (int i = 0; i < 10; i++) {
        std::cout << i << " " << array[i].throttle << " " << array[i].direction << std::endl;
    }

    std::cout << "Press Enter to exit...";
    std::cin.get();

    return 0;
}
