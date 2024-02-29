cmake_minimum_required(VERSION 3.15)
project(VideoPipeline)

set(CMAKE_CXX_STANDARD 11)

# OPTION(ENABLE_CUDA "使用CUDA,TensorRT" OFF)
# OPTION(ENABLE_TENSORRT "使用TensorRT" OFF)
# OPTION(ENABLE_HWCODEC "使用硬件编解码能力" OFF)


# 基础代码依赖配置
# include(${CMAKE_SOURCE_DIR}/cmakes/base.cmake)

# if (ENABLE_CUDA)
#     # CUDA依赖配置
#     include(${CMAKE_SOURCE_DIR}/cmake/cuda.cmake)
# endif ()
# if (ENABLE_TENSORRT)
#     # TensorRT依赖配置
#     include(${CMAKE_SOURCE_DIR}/cmake/tensorrt.cmake)
# endif ()


add_subdirectory(src)
# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(demo demo/main.cpp)

# 添加链接库目录
link_directories(${CMAKE_SOURCE_DIR}/libs)
target_link_libraries(demo zj_pipe)


