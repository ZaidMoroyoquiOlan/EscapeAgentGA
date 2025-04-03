#ifndef GENE_H
#define GENE_H

// Define export/import macros for Windows
#ifdef _WIN32
    #ifdef GENE_EXPORTS
        #define GENE_API __declspec(dllexport)
    #else
        #define GENE_API __declspec(dllimport)
    #endif
#else
    #define GENE_API __attribute__((visibility("default"))) // Linux/Mac
#endif

// Define the Gene struct
struct GENE_API Gene {
    float throttle;
    float direction;
};

#endif // GENE_H
