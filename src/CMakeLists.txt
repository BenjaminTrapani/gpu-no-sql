cmake_minimum_required(VERSION 3.3)
project(src)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    Makefile cudadriver.c.c cudadriver.c.h gpudbdriver.cu gpudbdriver.h DBCoreType.h QueryResult.h DBElement.h)

add_executable(src ${SOURCE_FILES})