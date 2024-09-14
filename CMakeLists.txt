cmake_minimum_required(VERSION 3.28)
project(dezero1)

set(CMAKE_CXX_STANDARD 23)

# Eigen library
find_package(Eigen3 REQUIRED)


add_executable(dezero1 main.cpp)

target_link_libraries(dezero1 Eigen3::Eigen)