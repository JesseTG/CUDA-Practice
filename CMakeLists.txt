cmake_minimum_required(VERSION 2.8)

project(CUDA-Practice)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/usr/local/share/cmake/Modules")
find_package(CUDA REQUIRED)

include_directories(Common)


add_subdirectory(Common)
add_subdirectory(Chapter3)
add_subdirectory(Chapter4)
add_subdirectory(Chapter5)
add_subdirectory(Chapter6)
add_subdirectory(Chapter7)
add_subdirectory(Chapter8)
add_subdirectory(Chapter9)

add_custom_target(CUDA-Practice-Parent
  SOURCES
    .gitignore
)
