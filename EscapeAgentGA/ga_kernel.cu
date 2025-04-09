#include <stdio.h>
#include <math.h>
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

__global__ void EvaluateChromosomeKernel(Gene* chromosomes, float* results, float* sourcePos, float* sourceVel, float heading, float* targetPos, int NChromosomes, int NActions) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    Gene* chromosome = &chromosomes[idx * NActions];
    float x = sourcePos[0], y = sourcePos[1];
    float vx = sourceVel[0], vy = sourceVel[1];
    float speed;
    float carAngle = heading;
    int currentGear;

    for (int i = 0; i < NActions; i++) {
        float throttle = chromosome[i].throttle; // Valor de [0 to 1.0]
        float direction = chromosome[i].direction; // Valor de [-1.0 to 1.0]

        // Girar el vehículo en radianes utilizando el modelo Angle-ratio
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

            // Actualizar velocidad basado en dirección
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

            // Actualizar posición
            x += vx * dt;
            y += vy * dt;
        }
    }
    // Distancia final a target
    float dx = x - targetPos[0];
    float dy = y - targetPos[1];
    results[idx] = sqrtf(dx * dx + dy * dy);
}

void generatePopulation(Gene *h_arr, int N, int M) {
    Gene *d_arr;
    size_t size = N * M * sizeof(Gene);

    cudaMalloc((void**) &d_arr, size);

    createRandomChromosome<<<1, N>>>(d_arr, N, M, time(NULL));
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

void EvaluateChromosomes(Gene* h_Chromosomes, float* h_Results, float* h_sourcePos, float* h_sourceVel, float heading, float *h_targetPos, int NChromosomes, int NActions) {
    Gene* d_Chromosomes;
    float* d_Results, *d_sourcePos, *d_sourceVel, *d_targetPos;
    size_t size = NChromosomes * NActions * sizeof(Gene);

    cudaMalloc((void**) &d_Chromosomes, size);
    cudaMalloc((void**) &d_Results, NChromosomes * sizeof(float));
    cudaMalloc((void**) &d_sourcePos, 2 * sizeof(float));
    cudaMalloc((void**) &d_sourceVel, 2 * sizeof(float));
    cudaMalloc((void**) &d_targetPos, 2 * sizeof(float));

    cudaMemcpy(d_Chromosomes, h_Chromosomes, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sourcePos, h_sourcePos, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sourceVel, h_sourceVel, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targetPos, h_targetPos, 2 * sizeof(float), cudaMemcpyHostToDevice);

    EvaluateChromosomeKernel<<<1, NChromosomes>>>(d_Chromosomes, d_Results, d_sourcePos, d_sourceVel, heading, d_targetPos, NChromosomes, NActions);
    cudaDeviceSynchronize();

    cudaMemcpy(h_Results, d_Results, NChromosomes * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Chromosomes);
    cudaFree(d_sourcePos);
    cudaFree(d_sourceVel);
    cudaFree(d_targetPos);
    cudaFree(d_Results);
}