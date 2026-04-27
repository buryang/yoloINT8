cmake_minimum_required(VERSION 3.16)
project(YOLOINT8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

set(NCNN_ROOT "${CMAKE_SOURCE_DIR}/tools/ncnn/ncnn-20260113-windows-vs2022")
set(NCNN_INCLUDE_DIR "${NCNN_ROOT}/x64/include")
set(NCNN_LIB_DIR "${NCNN_ROOT}/x64/lib")
set(NCNN_LIBRARY "${NCNN_LIB_DIR}/ncnn.lib")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${NCNN_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp
)

add_executable(yolo_inference
    cpp/main.cpp
)

target_link_libraries(yolo_inference
    ${OpenCV_LIBS}
    ${NCNN_LIBRARY}
)

target_compile_options(yolo_inference PRIVATE /W3)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "NCNN: ${NCNN_LIBRARY}")