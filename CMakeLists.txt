cmake_minimum_required(VERSION 3.20)
project(lob_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(lob_engine bindings.cpp)

target_include_directories(lob_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
