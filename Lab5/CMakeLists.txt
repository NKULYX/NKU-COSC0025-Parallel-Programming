cmake_minimum_required(VERSION 3.21)
project(Lab5)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXE_LINKER_FLAGS -static)

find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(x86 x86/Gauss.cpp)
add_executable(arm arm/Gauss.cpp)
add_executable(test test.cpp)

target_link_libraries(x86 ${MPI_LIBRARIES})
target_link_libraries(arm ${MPI_LIBRARIES})