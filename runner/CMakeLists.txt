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

# --- CUDA architecture targets -------------------------------------------
#
# Blackwell needs more than a plain compute capability. Per the PTX ISA
# guide (11.1.2 .target) and the Programming Guide (5.1.2):
#
#   sm_XY   baseline    onion-layer, runs on later generations
#   sm_XYf  family      runs on later devices in the SAME family only
#   sm_XYa  arch        runs ONLY on that exact compute capability
#
# tcgen05 (5th-gen tensor core) is NOT in the baseline set. It requires
# an f- or a-target, so a build for plain sm_100/sm_103 compiles happily
# and silently lacks the instructions the power virus depends on.
#
#   sm_100f  covers B200 (10.0) and B300 (10.3) — same sm_10x family.
#            Gets tcgen05.mma except a few .kind:: variants.
#   sm_103a  needed for the kinds excluded from family targets
#            (.kind::i8, .kind::mxf4, .kind::mxf4nvf4 — i.e. FP4 on B300).
#   sm_100a  same, for B200.
#   sm_120a  consumer/workstation Blackwell (5090, RTX PRO 6000). No
#            tcgen05; this is the T1 debug/validation target.
#
# We set these explicitly rather than via CMAKE_CUDA_ARCHITECTURES so the
# f/a suffixes work across CMake versions.
set(CMAKE_CUDA_ARCHITECTURES OFF)

set(GPL_CUDA_GENCODE
    -gencode arch=compute_120a,code=sm_120a
    -gencode arch=compute_100f,code=sm_100f
    -gencode arch=compute_100a,code=sm_100a
    -gencode arch=compute_103a,code=sm_103a
)

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
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall -Wextra -Wno-unused-parameter>
    $<$<AND:$<CONFIG:Release>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-O2>
    $<$<AND:$<CONFIG:Debug>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-O0 -g>
    # CUDA: emit every arch, and print per-kernel register/spill counts.
    # T0 gate parses this output — any "spill" line fails the build in CI,
    # because spilling turns the power virus from compute-bound into
    # memory-bound without changing anything visible in the source.
    $<$<COMPILE_LANGUAGE:CUDA>:${GPL_CUDA_GENCODE}>
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
)

install(TARGETS gpu-power-runner RUNTIME DESTINATION bin)
