cmake_minimum_required(VERSION 3.12)
project(ihmehimmeli)

include_directories(.)
include_directories(abseil-cpp)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(abseil-cpp)
add_subdirectory(common)
add_subdirectory(tempcoding)
add_subdirectory(event)
