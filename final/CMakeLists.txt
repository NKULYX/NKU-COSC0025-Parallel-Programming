cmake_minimum_required(VERSION 3.21)
project(final)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_EXE_LINKER_FLAGS -static)
set(CMAKE_CXX_FLAGS -march=native)

add_compile_options(-Wall)

find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(src)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(UnitTest test/UnitTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/x86/KMeansSIMD.cpp src/x86/KMeansPthread.cpp src/x86/KMeansOMP.cpp src/x86/KMeansMPI.cpp)
add_executable(x86SIMDTest test/x86/SIMDTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/x86/KMeansSIMD.cpp)
add_executable(x86PthreadTest test/x86/PthreadTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/x86/KMeansPthread.cpp)
add_executable(x86OMPTest test/x86/OMPTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/x86/KMeansOMP.cpp)
add_executable(x86MPITest test/x86/MPITest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/x86/KMeansMPI.cpp)
add_executable(armSIMDTest test/arm/SIMDTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/arm/KMeansSIMD.cpp)
add_executable(armPthreadTest test/arm/PthreadTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/arm/KMeansPthread.cpp)
add_executable(armOMPTest test/arm/OMPTest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/arm/KMeansOMP.cpp)
add_executable(armMPITest test/arm/MPITest.cpp src/KMeans.cpp src/TestUtil.cpp src/KMeansSerial.cpp src/arm/KMeansMPI.cpp)

target_link_libraries(x86MPITest ${MPI_LIBRARIES})
