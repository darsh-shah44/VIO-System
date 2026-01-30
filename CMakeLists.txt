cmake_minimum_required(VERSION 3.10)
project(VIO)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

add_executable(vio src/main.cpp)
target_link_libraries(vio ${OpenCV_LIBS} Eigen3::Eigen)