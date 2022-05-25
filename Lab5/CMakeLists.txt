cmake_minimum_required(VERSION 3.21)
project(Lab5)

set(CMAKE_CXX_STANDARD 14)

find_package(MPI REQUIRED)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(Lab5 main.cpp)

target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})