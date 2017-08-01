cmake_minimum_required(VERSION 3.0)

project(bitplanes-tracking)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

find_package(OpenCV REQUIRED)
find_package(Eigen REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_subdirectory(src)
