cmake_minimum_required(VERSION 3.7)
project(DatasetGeneration)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

if (WIN32)
    set(OpenCV_DIR "D:\\opencv\\mingw_build\\install")
endif ()

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_MODULE_DIR}/cmake/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if (WIN32)
    set(Eigen3_DIR "D:\\eigen3\\cmake")
endif ()

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(main main.cpp DatasetGenerator.cpp)
add_executable(main_test test.cpp DatasetGenerator.cpp)
target_link_libraries(main Eigen3::Eigen ${OpenCV_LIBS})
target_link_libraries(main_test Eigen3::Eigen ${OpenCV_LIBS})