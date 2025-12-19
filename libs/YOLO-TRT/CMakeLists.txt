cmake_minimum_required(VERSION 3.16)
project(YOLO-TRT LANGUAGES CXX C)

# 设置 C++ 17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

message(">> Configure YOLO-TRT")

add_library(${PROJECT_NAME} 
    ${LIB_TYPE} 
        yolo_trt.cpp 
        yolo_trt_cls.cpp
        yolo_trt_det.cpp
        yolo_trt_obb.cpp
        yolo_trt_pose.cpp
        yolo_trt_seg.cpp)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        ${OpenCV_LIBS}
        TRTInfer
)