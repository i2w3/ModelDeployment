cmake_minimum_required(VERSION 3.16)
project(ModelDeployment LANGUAGES CXX C)

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

add_compile_definitions(
    # 如果是 Debug 模式，定义 IS_DEBUG
    $<$<CONFIG:Debug>:IS_DEBUG>
    
    # 如果是 Release 模式，定义 IS_RELEASE
    $<$<CONFIG:Release>:IS_RELEASE>
)

# 设置指定项目生成的可执行文件路径
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin/Debug)
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin/Release)
endif()

set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/libs/opencv")
set(OPENCV_BIN_DIR "${CMAKE_SOURCE_DIR}/libs/opencv/x64/vc16/bin")

set(spdlog_DIR "${CMAKE_SOURCE_DIR}/libs/spdlog/build")

find_package(CUDAToolkit REQUIRED) # for TRTInfer CUDA::cudart
find_package(OpenCV CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)


add_subdirectory(libs/PaddleOCR)
add_subdirectory(libs/TRTInfer)
add_subdirectory(libs/YOLO)

add_executable(main src/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})

add_executable(test_yolo src/test_yolo.cpp)
target_link_libraries(test_yolo ${OpenCV_LIBS} YOLO)

# add_executable(test_det src/test_det.cpp)
# target_link_libraries(test_det ${OpenCV_LIBS} PaddleOCR)

# add_executable(test_cls src/test_cls.cpp)
# target_link_libraries(test_cls ${OpenCV_LIBS} PaddleOCR)

# 复制对应配置的 world DLL
add_custom_command(TARGET main POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy
        "$<IF:$<CONFIG:Debug>,${OPENCV_BIN_DIR}/opencv_world4120d.dll,${OPENCV_BIN_DIR}/opencv_world4120.dll>"
        "$<TARGET_FILE_DIR:main>"
)

# 复制 ffmpeg DLL（两边都一样）
add_custom_command(TARGET main POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${OPENCV_BIN_DIR}/opencv_videoio_ffmpeg4120_64.dll"
        "$<TARGET_FILE_DIR:main>"
)