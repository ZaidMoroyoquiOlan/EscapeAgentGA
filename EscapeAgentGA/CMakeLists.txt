cmake_minimum_required(VERSION 3.18)
project(EscapeAgentGA LANGUAGES CXX CUDA)

# Create shared library (DLL) including CUDA source
add_library(EscapeAgentGA SHARED EscapeAgentGA.cpp ga_kernel.cu)

include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/include")

# Set C++ and CUDA standards
set_target_properties(EscapeAgentGA PROPERTIES 
    CXX_STANDARD 20
    CUDA_STANDARD 17
)

# Define export macro for Windows
target_compile_definitions(EscapeAgentGA PRIVATE ESCAPEAGENTGA_EXPORTS)

# Ensure CUDA separable compilation
set_target_properties(EscapeAgentGA PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
)

# Set output directories
set_target_properties(EscapeAgentGA PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Create test executable
add_executable(EscapeAgentGATest test_main.cpp)

# Ensure test executable is placed in bin
set_target_properties(EscapeAgentGATest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# **LINK THE TEST EXECUTABLE TO THE DLL**
target_link_libraries(EscapeAgentGATest PRIVATE EscapeAgentGA)
