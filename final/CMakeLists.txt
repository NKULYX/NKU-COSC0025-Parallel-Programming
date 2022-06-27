cmake_minimum_required(VERSION 3.21)
project(final)

include_directories(src)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

add_compile_options(-Wall)

add_executable(UnitTest test/UnitTest.cpp src/KMeansSerial.cpp src/KMeansCUDA.cu src/KMeans.cpp src/TestUtil.cpp)
add_executable(SIMDTest src/KMeansSerial.cpp test/SIMDTest.cpp)
add_executable(CUDATest test/CUDATest.cpp src/KMeansCuda.cu)

set_target_properties(CUDATest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
