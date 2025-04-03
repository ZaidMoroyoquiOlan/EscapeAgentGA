// EscapeAgentGA.h : Include file for standard system include files,
// or project specific include files.

#pragma once
#include "gene.h"

#ifdef _WIN32
    #ifdef ESCAPEAGENTGA_EXPORTS
        #define ESCAPEAGENTGA_API __declspec(dllexport)
    #else
        #define ESCAPEAGENTGA_API __declspec(dllimport)
    #endif
#else
    #define ESCAPEAGENTGA_API
#endif

extern "C" {
    ESCAPEAGENTGA_API void RunCUDAGA(Gene *solution);  // Function exposed by the DLL
}