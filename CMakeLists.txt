cmake_minimum_required(VERSION 3.15)
project(ViolaJonesDetector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(ViolaJonesDetector
    src/main.cpp
)
include_directories(include)
