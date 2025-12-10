cmake_minimum_required(VERSION 3.20)
project(TRTInfer LANGUAGES CXX CUDA VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    # 增加 UTF-8 支持
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    # 严格检查
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()

set(LIB_TYPE SHARED)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_TYPE STATIC)
endif()

message(">> Configure TRTInfer")

set(TensorRT_INCLUDE_DIRS 
    "$ENV{TensorRT_INCLUDE_DIR}")
if(WIN32)
    set(TensorRT_LIBS
        "$ENV{TensorRT_LIBRARY_PATH}/nvinfer_10.lib"
        "$ENV{TensorRT_LIBRARY_PATH}/nvinfer_plugin_10.lib"
    )
else()
    set(TensorRT_LIBS
        "$ENV{TensorRT_LIBRARY_PATH}/libnvinfer.so"
        "$ENV{TensorRT_LIBRARY_PATH}/libnvinfer_plugin.so"
    )
endif()

add_library(TRTInfer ${LIB_TYPE}
    TRTInfer.cc
    utility.cc
)

target_include_directories(TRTInfer
    SYSTEM PUBLIC
        ${TensorRT_INCLUDE_DIRS}
    PUBLIC
        ${CMAKE_BINARY_DIR}/libs/TRTInfer
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(TRTInfer 
    PUBLIC
        CUDA::cudart
        spdlog::spdlog
        ${OpenCV_LIBS}
        ${TensorRT_LIBS}
)