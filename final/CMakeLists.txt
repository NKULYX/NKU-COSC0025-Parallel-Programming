cmake_minimum_required(VERSION 3.21)
project(final)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(-Wall)

add_executable(main main.cpp KMeans.cpp)
add_executable(SIMD  KMeans.cpp SIMDTest.cpp)
