cmake_minimum_required(VERSION 3.22)
project(fft_ced LANGUAGES CXX CUDA)

set(CMAKE_CXX_FLAGS "-fopenmp -O2 -std=c++17")
set(CMAKE_CUDA_FLAGS "-O2")

include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(fft
    src/fft.cpp
    src/fft_gpu.cu
    src/main.cpp
)

target_link_libraries(fft 
    ${CUDA_LIBRARIES}
    ${CUDA_CUFFT_LIBRARIES}
)

