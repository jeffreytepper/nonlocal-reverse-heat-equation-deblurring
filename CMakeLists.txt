cmake_minimum_required(VERSION 3.3.2)

project(my_hpx_project CXX)

find_package(HPX REQUIRED)

add_hpx_executable(serial SOURCES NLRHEQ_sequential.cpp)

#add_hpx_executable(parallel SOURCES NLRHEQ_parallel.cpp)
