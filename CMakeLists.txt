cmake_minimum_required(VERSION 3.10)
project(VideoPipeline)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmakes)
#设置宏
add_definitions(-DEnable_ONNX)

set(CMAKE_CXX_STANDARD 11)

find_package(Threads REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(FFmpeg REQUIRED)


# 添加头文件路径与库文件
include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIBRARIES})


OPTION(ENABLE_CUDA "使用CUDA,TensorRT" ON)
OPTION(ENABLE_TENSORRT "使用TensorRT" ON)
# OPTION(ENABLE_HWCODEC "使用硬件编解码能力" OFF)

if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV")
endif()


include_directories(${OpenCV_INCLUDE_DIRS})


# find ffmpeg

# 基础代码依赖配置
# include(${CMAKE_SOURCE_DIR}/cmakes/base.cmake)

if (ENABLE_CUDA)
    # CUDA依赖配置
    add_definitions(-DEnable_CUDA)
    include(${CMAKE_SOURCE_DIR}/cmakes/cuda.cmake)

endif ()
if (ENABLE_TENSORRT)
    # TensorRT依赖配置
    include(${CMAKE_SOURCE_DIR}/cmakes/tensorrt.cmake)
    add_definitions(-DEnable_TRT)
endif ()


add_subdirectory(src)
# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

add_executable(demo demo/main.cpp)

# 添加链接库目录
link_directories(${CMAKE_SOURCE_DIR}/libs)
# link_directories(/home/wnk/code/software/onnxruntime-linux-x64-1.17.1/lib)
# target_link_libraries(demo onnxruntime)

target_link_libraries(demo zj_pipe cuda_lib ${CMAKE_SOURCE_DIR}/libs/libonnxruntime.so pthread ${X11_LIBRARIES} ${OpenCV_LIBS} ${TRT_LIBS} ${CUDA_LIBS} ${CUDNN_LIBS})