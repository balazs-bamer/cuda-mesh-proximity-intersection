cmake_minimum_required(VERSION 3.16)
project(IntersectionReference)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-ggdb)
INCLUDE_DIRECTORIES ( "/usr/include/eigen3" )

add_executable(IntersectionReference IntersectionReference.cpp)
