// EscapeAgentGA.h : Include file for standard system include files,
// or project specific include files.

#pragma once
#include "gene.h"
#include <stdint.h>

#ifdef _WIN32
    #ifdef ESCAPEAGENTGA_EXPORTS
        #define ESCAPEAGENTGA_API __declspec(dllexport)
    #else
        #define ESCAPEAGENTGA_API __declspec(dllimport)
    #endif
#else
    #define ESCAPEAGENTGA_API
#endif

// Functions exposed by the DLL
extern "C" {
    ESCAPEAGENTGA_API void InitCudaArray(const uint8_t* binArray, size_t size);
    ESCAPEAGENTGA_API void RunCUDAGA(Gene* solution, float* sourcePos, float* sourceVel, float heading, float* targetPos, float targetHeading, int m);
    ESCAPEAGENTGA_API void CleanupCuda();
}