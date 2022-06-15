cmake_minimum_required(VERSION 3.21)
project(final)

include_directories(src)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

add_compile_options(-Wall)

add_executable(test test.cpp src/KMeans.cpp src/KMeansCUDA.cu)
add_executable(SIMD src/KMeans.cpp SIMDTest.cpp)

set_target_properties(test PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
