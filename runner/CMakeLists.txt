cmake_minimum_required(VERSION 3.20)
project(gpu_power_runner LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# NVML ships in the driver, but the stub header + import lib come with the toolkit.
# CUDAToolkit provides CUDA::nvml.

add_executable(gpu-power-runner
    src/main.c
    src/args.c
    src/telemetry.c
    src/workload_gemm.c
    src/workload_memstream.cu
    src/workload_fft.c
    src/workload_util.c
    src/summary.c
    src/util.c
)

target_include_directories(gpu-power-runner PRIVATE src)

target_link_libraries(gpu-power-runner PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cufft
    CUDA::nvml
    m
    pthread
)

target_compile_options(gpu-power-runner PRIVATE
    -Wall -Wextra -Wno-unused-parameter
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:Debug>:-O0 -g>
)

install(TARGETS gpu-power-runner RUNTIME DESTINATION bin)
