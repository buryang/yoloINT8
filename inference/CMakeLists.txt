cmake_minimum_required(VERSION 3.16)
project(YOLOInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
set(OpenCV_DIR "D:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

# NCNN
set(NCNN_DIR "D:/ncnn/ncnn-20260113-windows-vs2022/x64")
set(NCNN_INCLUDE_DIR "${NCNN_DIR}/include")
set(NCNN_LIB_DIR "${NCNN_DIR}/lib")
include_directories(${NCNN_INCLUDE_DIR})
link_directories(${NCNN_LIB_DIR})

# Add executable
add_executable(yolo_test main.cpp)

# Link libraries
target_link_libraries(yolo_test ${OpenCV_LIBS})
target_link_libraries(yolo_test ncnn)
target_link_libraries(yolo_test glslang)
target_link_libraries(yolo_test SPIRV)

# Windows specific
if(WIN32)
    target_compile_options(yolo_test PRIVATE /W0)
endif()