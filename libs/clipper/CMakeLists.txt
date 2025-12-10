cmake_minimum_required(VERSION 3.16)
project(clipper LANGUAGES CXX C)

set(LIB_TYPE SHARED)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_TYPE STATIC)
endif()

message(">> Configure clipper")

add_library(${PROJECT_NAME} ${LIB_TYPE} clipper.cpp)
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)