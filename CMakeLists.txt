cmake_minimum_required(VERSION 3.10)

# プロジェクト名とバージョン
project(DroneOptimization VERSION 1.0)

# C++の標準を設定
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 実行ファイルの出力先をレポジトリのトップディレクトリに設定
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# インクルードディレクトリの設定
include_directories(${PROJECT_SOURCE_DIR}/libs)
include_directories(${PROJECT_SOURCE_DIR}/3rd)

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

# ソースファイルのリスト
file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

# 実行ファイルを作成
add_executable(DroneOptimization ${SOURCES})

# リンクするライブラリ
target_link_libraries(DroneOptimization ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})

# リソースファイルをコピー
file(COPY ${PROJECT_SOURCE_DIR}/resource DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# テストの設定
file(GLOB TEST_SOURCES ${PROJECT_SOURCE_DIR}/test/*.cpp)

# テスト用のソースファイルがあるか確認
if(TEST_SOURCES)
    enable_testing()
    add_executable(DroneOptimizationTests ${TEST_SOURCES})
    target_link_libraries(DroneOptimizationTests ${OPENGL_LIBRARIES} ${GLUT_glut_LIBRARY})

    # テストを登録
    add_test(NAME DroneOptimizationTests COMMAND DroneOptimizationTests)
else()
    message(STATUS "No test files found. Skipping test build.")
endif()
