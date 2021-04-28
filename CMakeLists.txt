cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

project(TutorialSamples LANGUAGES CUDA CXX)

add_subdirectory(01_HelloGPU)
add_subdirectory(02_HostDeviceFunctions)
add_subdirectory(03_GridConfiguration)
add_subdirectory(04_LegacyScheduling)
add_subdirectory(05_IndependentThreadScheduling)
add_subdirectory(06_MemoryBasics)
add_subdirectory(07_Reductions)

