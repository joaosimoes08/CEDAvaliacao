cmake_minimum_required(VERSION 3.22)
project(fft_ced LANGUAGES CXX CUDA)
find_package(CUDAToolkit)

set(CMAKE_CXX_FLAGS "-fopenmp -O2 -std=c++17")
set(CMAKE_CUDA_FLAGS "-O2")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(fft_gpu
    src/fft_gpu.cu
    src/fft.cpp
    src/main.cpp
)
