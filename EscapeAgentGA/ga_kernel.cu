#include <stdio.h>
#include <curand_kernel.h>
#include "ga_kernel.cuh"
#include "gene.h"

__global__ void createRandomChromosome(Gene* arr, int N, int M, int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N) return;

    // Calculate the starting index for each thread's array
    Gene* threadArray = arr + idx * M;

    // Initialize cuRAND state for each thread
    curandState state;
    curand_init(seed, idx, 0, &state);

    for (int i = 0; i < M; i++) {
        threadArray[i].direction = 2.0f * curand_uniform(&state) - 1.0f;
        threadArray[i].throttle = curand_uniform(&state);
    }
}

__global__ void myKernel() {
    printf("Hello from CUDA Kernel!\n");
}

void generatePopulation(Gene *h_arr, int N, int M) {
    Gene *d_arr;
    size_t size = N * M * sizeof(Gene);

    cudaMalloc((void**) &d_arr, size);

    createRandomChromosome<<<1, 1>>>(d_arr, N, M, time(NULL));
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}