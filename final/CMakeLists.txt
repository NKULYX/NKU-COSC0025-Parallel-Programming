cmake_minimum_required(VERSION 3.21)
project(final)

include_directories(src)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_EXE_LINKER_FLAGS -static)
set(CMAKE_CXX_FLAGS -march=native)

add_compile_options(-Wall)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(UnitTest test/UnitTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/KMeansSIMD.cpp src/KMeansPthread.cpp src/KMeansOMP.cpp)
add_executable(SIMDTest test/SIMDTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/KMeansSIMD.cpp)
add_executable(PthreadTest test/PthreadTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/KMeansPthread.cpp)
add_executable(OMPTest test/OMPTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/KMeansOMP.cpp)
add_executable(CUDATest test/CUDATest.cpp src/KMeansCuda.cu)

set_target_properties(CUDATest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
