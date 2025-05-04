cmake_minimum_required(VERSION 3.30)
project(lab1_mpi_det)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 20)

add_executable(lab1_mpi_det main.cpp)
target_link_libraries(lab1_mpi_det MPI::MPI_CXX)
