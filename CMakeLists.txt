cmake_minimum_required(VERSION 3.18)

# プロジェクト名とバージョン
project(DroneOptimization VERSION 1.0 LANGUAGES CXX CUDA)

# C++の標準を設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fopenmp")

# 実行ファイルの出力先
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# ライブラリの出力先
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/libs)

# インクルードディレクトリの設定
include_directories(${PROJECT_SOURCE_DIR}/libs)
include_directories(${PROJECT_SOURCE_DIR}/3rd)
include_directories(/usr/local/cuda/include)

# OpenGLおよびGLUTのライブラリを検索
find_package(OpenGL REQUIRED)

# GLUTのインクルードディレクトリとライブラリを手動で指定
set(GLUT_INCLUDE_DIR "/usr/include/GL")
set(GLUT_glut_LIBRARY "/usr/lib/x86_64-linux-gnu/libglut.so")

if (NOT EXISTS ${GLUT_INCLUDE_DIR}/glut.h OR NOT EXISTS ${GLUT_glut_LIBRARY})
    message(FATAL_ERROR "GLUT not found!")
else()
    include_directories(${GLUT_INCLUDE_DIR})
endif()

# CUDA関連のソースコードを追加
aux_source_directory(${PROJECT_SOURCE_DIR}/src/Cuda Cuda_SOURCES)

# CUDAライブラリの設定
add_library(cudaUtil SHARED ${Cuda_SOURCES})
link_directories(/usr/local/cuda/lib64)
target_link_libraries(cudaUtil libcudart.so)
set_target_properties(cudaUtil PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
    CUDA_ARCHITECTURES 52)
target_compile_options(cudaUtil PRIVATE -fopenmp)

# src/Main/ の中の各 .cpp ファイルごとに実行ファイルを作成
file(GLOB MAIN_SOURCES ${PROJECT_SOURCE_DIR}/src/Main/*.cpp)
foreach(MAIN_SOURCE ${MAIN_SOURCES})
    get_filename_component(EXECUTABLE_NAME ${MAIN_SOURCE} NAME_WE)
    add_executable(${EXECUTABLE_NAME} ${MAIN_SOURCE})
    target_link_libraries(${EXECUTABLE_NAME} ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} cudaUtil)
endforeach()

# テストの設定
file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/*.cpp)

if(TEST_SOURCES)
    enable_testing()
    add_executable(DroneOptimizationTests ${TEST_SOURCES})
    target_link_libraries(DroneOptimizationTests ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY} cudaUtil)

    # テストを登録
    add_test(NAME DroneOptimizationTests COMMAND DroneOptimizationTests)
else()
    message(STATUS "No test files found. Skipping test build.")
endif()

# リソースファイルをコピー
file(COPY ${PROJECT_SOURCE_DIR}/resource DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
