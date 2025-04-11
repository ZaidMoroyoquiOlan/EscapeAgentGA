#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>
#include "ga_kernel.cuh"
#include "gene.h"

#define PI 3.14159265359f
#define wheel_base 250.0f     // cm
#define max_speed 3000.0f     // cm/s
#define max_steering_deg 45.0f
#define max_torque 300.0f     // N
#define tire_radius 0.36f     // m
#define dt 0.1f
#define steps_per_second (int)(1.0f / dt)

__constant__ float gearRatios[7] = { 3.75, 2.38, 1.72, 1.34, 1.11, 0.96, 0.84 };
__constant__ float changeRatio[7] = { 0, 8.05, 13.33, 17.22, 18.88, 26.11, 30.0 };

__device__ int GetCurrentGear(float speed, int numGears) {
    for (int i = 0; i < numGears; i++) {
        if (speed < changeRatio[i + 1]) {
            return i;
        }
    }
    return 0;
}

__device__ float GetTorqueFromRPM(float rpm, float maxTorque, float maxRPM) {
    float peakRPM = maxRPM * 0.7f;
    float normalized = (rpm - peakRPM) / (maxRPM * 0.45f);
    float torqueFactor = expf(-normalized * normalized);
    return maxTorque * torqueFactor;
}

__device__ float GetEngineRPM(float speed, float gearRatio, float finalDriveRatio, float wheelRadius) {
    float wheelRPM = (speed / (2 * PI * wheelRadius)) * 60.0f;
    return wheelRPM * gearRatio * finalDriveRatio;
}

__device__ float GetWheelTorque(float engineTorque, float gearRatio, float finalDriveRatio) {
    return engineTorque * gearRatio * finalDriveRatio * 0.9f;
}

__global__ void EvaluateChromosome(Gene* chromosomes, float* fitness, float* sourcePos, float* sourceVel, float heading, float* targetPos, float targetHeading, int NChromosomes, int NActions) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NChromosomes) return;
    
    Gene* chromosome = &chromosomes[idx * NActions];
    float x = sourcePos[0], y = sourcePos[1];
    float vx = sourceVel[0], vy = sourceVel[1];
    float speed;
    float carAngle = heading;
    int currentGear;

    for (int i = 0; i < NActions; i++) {
        float throttle = chromosome[i].throttle; // Valor de [0 to 1.0]
        float direction = chromosome[i].direction; // Valor de [-1.0 to 1.0]

        // Girar el veh�culo en radianes utilizando el modelo Angle-ratio
        float steeringAngle = 0.7f * direction * (3.14159265359f / 180.0f) * max_steering_deg;

        for (int j = 0; j < steps_per_second; j++) {
            speed = sqrtf(vx * vx + vy * vy);
            currentGear = GetCurrentGear(speed / 100.0, 7);

            float engineRPM = GetEngineRPM(speed / 100.0, gearRatios[currentGear], 3.97, tire_radius);
            float engineTorque = GetTorqueFromRPM(engineRPM, max_torque, 4500.0f);
            float torque = GetWheelTorque(engineTorque, gearRatios[currentGear], 3.97);

            float driveForce = (torque * throttle) / tire_radius; // N

            float acceleration = 100 * driveForce / 1500.0f; // cm/s^2

            float dv = acceleration * dt;

            // Actualizar velocidad basado en direcci�n
            if (fabsf(steeringAngle) < 0.001f) {
                // Move straight
                vx += dv * cosf(carAngle);
                vy += dv * sinf(carAngle);
            }
            else {
                // Girar utilizando el modelo de bicicleta
                float turningRadius = wheel_base / tanf(steeringAngle);
                float angularVelocity = speed / turningRadius; // radians/sec
                carAngle += angularVelocity * dt;

                vx += dv * cosf(carAngle);
                vy += dv * sinf(carAngle);
            }

            // Actualizar posici�n
            x += vx * dt;
            y += vy * dt;
        }
    }
    // Facing
    float dx = targetPos[0] - x;
    float dy = targetPos[1] - y;
    float toTargetMag = sqrtf(dx * dx + dy * dy);

    if (toTargetMag > 1e-5f) {
        dx /= toTargetMag;
        dy /= toTargetMag;
    }

    float facingX = cosf(carAngle);
    float facingY = sinf(carAngle);

    float tFacingX = cosf(carAngle);
    float tFacingY = sinf(carAngle);

    // Looking at another direction away from where target is facing
    float dot = fmaxf(-1.0f, fminf(1.0f, tFacingX * facingX + tFacingY * facingY));
    float angleFactor = acosf(dot) / 3.14159265f;

    // Looking at another direction away from where target is at
    dot = fmaxf(-1.0f, fminf(1.0f, dx * facingX + dy * facingY));
    float angleLocationFactor = acosf(dot) / 3.14159265f;

    fitness[idx] = angleFactor*(toTargetMag / 100.0f) + 30*angleLocationFactor;
}

__global__ void InitializePopulation(Gene* d_Chromosomes, curandState* states, int NChromosomes, int NActions) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NChromosomes) {
        curandState localState = states[idx];

        for (int i = 0; i < NActions; i++) {
            d_Chromosomes[idx * NActions + i].throttle = curand_uniform(&localState); // [0,1]
            d_Chromosomes[idx * NActions + i].direction = curand_uniform(&localState) * 2.0f - 1.0f; // [-1,1]
        }
        states[idx] = localState;
    }
}

__global__ void InitRandomStates(curandState* states, unsigned long seed, int NChromosomes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < NChromosomes) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__device__ int Ranking(Gene* current_genes, Gene* new_genes, int chromo_index, float* fitness, curandState* state, int num_chromosomes, int num_genes) {
    int r1 = curand(state) % num_chromosomes;
    int r2;

    for (int i = 0; i < (num_chromosomes / 2); i++) {
        r2 = curand(state) % num_chromosomes;
        if (fitness[r1] < fitness[r2]) {
            r1 = r2;
        }
    }
    for (int i = 0; i < num_genes; i++) {
        new_genes[chromo_index + i] = current_genes[r1 + i];
    }

    return r1;
}

__device__ void Crossover(Gene* current_genes, Gene* new_genes, int parent1, int parent2, int base_index, int split, int NActions) {
    for (int i = 0; i < NActions; i++) {
        new_genes[base_index + i] = (i < split) ? current_genes[parent1 + i] : current_genes[parent2 + i];
    }
}

__device__ void Mutate(Gene* genes, int chromo_index, float mutation_rate, curandState* state, int NActions) {
    for (int i = 0; i < NActions; i++) {
        if (curand_uniform(state) < mutation_rate) {
            genes[chromo_index + i].throttle = curand_uniform(state);
            genes[chromo_index + i].direction = curand_uniform(state) * 2.0f - 1.0f;
        }
    }
}

__global__ void Recombination(Gene* d_Population, Gene* d_NewPopulation, float* fitness, curandState* states, int NChromosomes, int NActions) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < NChromosomes / 4) {
        curandState localState = states[idx];

        int base = idx * 4;

        // Los padres son insertados directamente en la nueva generaci�n
        int parent1 = Ranking(d_Population, d_NewPopulation, base, fitness, &localState, NChromosomes, NActions);
        int parent2 = Ranking(d_Population, d_NewPopulation, base + NActions, fitness, &localState, NChromosomes, NActions);

        // Los hijos son insertados directamente en la nueva generaci�n
        int split = curand(&localState) % NActions;
        Crossover(d_Population, d_NewPopulation, parent1, parent2, base + 2 * NActions, split, NActions);
        Crossover(d_Population, d_NewPopulation, parent2, parent1, base + 3 * NActions, split, NActions);

        Mutate(d_NewPopulation, base + 2 * NActions, 0.1f, &localState, NActions);
        Mutate(d_NewPopulation, base + 3 * NActions, 0.1f, &localState, NActions);

        states[idx] = localState;
    }
}

__global__ void getBestSolution(Gene* chromosomes, float* fitness, Gene* d_bestSolution, float *bestFitness, int NChromosomes, int NActions) {
    int best = 0; 

    for (int i = 1; i < NChromosomes; i++) {
        if (fitness[i] > fitness[best]) {
            best = i;
        }
    }

    if (fitness[best] > *bestFitness) {
        *bestFitness = fitness[best];

        for (int i = 0; i < NActions; i++) {
            d_bestSolution[i] = chromosomes[best*NActions + i];
        }
    }
    printf("Best solution[%d]: %f", best, *bestFitness);
}

void ExecuteGA(Gene* solution, float* sourcePos, float* sourceVel, float heading, float* targetPos, float targetHeading, int NChromosomes, int NActions, int NGenerations) {
    size_t populationSize = NChromosomes * NActions * sizeof(Gene);

    Gene* d_Population, *d_NewPopulation, *d_BestSolution;
    cudaMalloc((void **) &d_Population, populationSize);
    cudaMalloc((void **) &d_NewPopulation, populationSize);
    cudaMalloc((void**) &d_BestSolution, NActions * sizeof(Gene));

    float* fitness, *bestFitness, h_bestFitness = 0.0f;
    float* d_sourcePos, *d_sourceVel, *d_targetPos;
    cudaMalloc((void**) &fitness, NChromosomes * sizeof(float));
    cudaMalloc((void**) &bestFitness, sizeof(float));
    cudaMalloc((void**) &d_sourcePos, 2*sizeof(float));
    cudaMalloc((void**) &d_sourceVel, 2*sizeof(float));
    cudaMalloc((void**) &d_targetPos, 2*sizeof(float));

    cudaMemcpy(bestFitness, &h_bestFitness, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sourcePos, sourcePos, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sourceVel, sourceVel, 2*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targetPos, targetPos, 2*sizeof(float), cudaMemcpyHostToDevice);

    curandState* states;
    cudaMalloc(&states, NChromosomes * sizeof(curandState));
    InitRandomStates<<<1, NChromosomes>>>(states, time(NULL), NChromosomes);
    cudaDeviceSynchronize();

    InitializePopulation<<<1, NChromosomes>>>(d_Population, states, NChromosomes, NActions);
    cudaDeviceSynchronize();

    EvaluateChromosome<<<1, NChromosomes>>>(d_Population, fitness, d_sourcePos, d_sourceVel, heading, d_targetPos, targetHeading, NChromosomes, NActions);
    cudaDeviceSynchronize();

    getBestSolution<<<1,1>>>(d_Population, fitness, d_BestSolution, bestFitness, NChromosomes, NActions);
    cudaDeviceSynchronize();

    for (int g = 0; g < NGenerations; g++) {
        Recombination<<<1, NChromosomes>>>(d_Population, d_NewPopulation, fitness, states, NChromosomes, NActions);
        cudaDeviceSynchronize();

        cudaMemcpy(d_Population, d_NewPopulation, populationSize, cudaMemcpyDeviceToDevice);

        EvaluateChromosome<<<1, NChromosomes>>>(d_Population, fitness, d_sourcePos, d_sourceVel, heading, d_targetPos, targetHeading, NChromosomes, NActions);
        cudaDeviceSynchronize();

        getBestSolution<<<1,1>>>(d_Population, fitness, d_BestSolution, bestFitness, NChromosomes, NActions);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(solution, d_BestSolution, NActions * sizeof(Gene), cudaMemcpyDeviceToHost);

    cudaFree(d_Population);
    cudaFree(d_NewPopulation);
    cudaFree(d_BestSolution);
    cudaFree(d_sourcePos);
    cudaFree(d_sourceVel);
    cudaFree(d_targetPos);
    cudaFree(bestFitness);
    cudaFree(fitness);
    cudaFree(states);
}