cmake_minimum_required(VERSION 3.10)
project(DatasetGeneration)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (Eigen3 3.3 NO_MODULE)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

add_executable(main main.cpp DatasetGenerator.cpp)
add_executable(test test.cpp DatasetGenerator.cpp)
target_link_libraries(main Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(test Eigen3::Eigen ${OpenCV_LIBS})