cmake_minimum_required(VERSION 3.22)
project(fft_ced LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-fopenmp -O2 -std=c++17")

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(fft
    src/fft.cpp
    src/main.cpp
)
