cmake_minimum_required(VERSION 3.21)
project(Lab5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS -static)
set(CMAKE_CXX_FLAGS -march=native)

find_package(MPI REQUIRED)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(x86 x86/Gauss.cpp)
add_executable(arm arm/Gauss.cpp)
add_executable(test test.cpp)

target_link_libraries(x86 ${MPI_LIBRARIES})
target_link_libraries(arm ${MPI_LIBRARIES})