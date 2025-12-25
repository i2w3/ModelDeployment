cmake_minimum_required(VERSION 3.16)
project(ModelDeployment LANGUAGES CXX CUDA)

# 设置 C++ 17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置指定项目生成的可执行文件路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 查找所需的包
find_package(spdlog CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED) # for TRTInfer CUDA::cudart

# 传递给子项目 libs/TensorRT-YOLO
set(TRT_PATH "/usr/lib/x86_64-linux-gnu" CACHE PATH "Path to TensorRT installation")

# add_subdirectory(libs/clipper)
add_subdirectory(libs/ModelEncrypt)
# add_subdirectory(libs/PaddleOCR)
add_subdirectory(libs/TensorRT-YOLO)
add_subdirectory(libs/TRTInfer)
add_subdirectory(libs/YOLO-TRT)

if(MSVC)
    # 增加 UTF-8 支持
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    # 严格检查
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra)
endif()

add_compile_definitions(
    # 如果是 Debug 模式，定义 IS_DEBUG
    $<$<CONFIG:Debug>:IS_DEBUG>
    
    # 如果是 Release 模式，定义 IS_RELEASE
    $<$<CONFIG:Release>:IS_RELEASE>
)

add_library(include_folder INTERFACE)
target_include_directories(include_folder INTERFACE ./include)
target_include_directories(include_folder SYSTEM INTERFACE
    $<TARGET_PROPERTY:YOLO-TRT,INTERFACE_INCLUDE_DIRECTORIES>
)

set(TEST_TARGETS
    test_rtsp_gst
    test_rtsp_gst_pc
    test_rtsp_ffmpeg
    test_yolo_trt
    test_yolo_trt_gst
)

foreach(target ${TEST_TARGETS})
    add_executable(${target} ./src/${target}.cpp)
    target_link_libraries(${target} PRIVATE ${OpenCV_LIBS} include_folder)
endforeach()

# 为特定目标添加额外依赖
target_link_libraries(test_yolo_trt PRIVATE YOLO-TRT)
target_link_libraries(test_yolo_trt_gst PRIVATE YOLO-TRT)

add_executable(test_encrypt ./src/test_encrypt.cpp)
target_link_libraries(test_encrypt PRIVATE ModelEncrypt)

add_executable(test_other_yolo_trt ./src/test_other_yolo_trt.cpp)
target_include_directories(test_other_yolo_trt PRIVATE
    libs/TensorRT-YOLO/modules/trtyolo/infer
)
target_link_libraries(test_other_yolo_trt PRIVATE 
    ${OpenCV_LIBS}
    trtyolo
)