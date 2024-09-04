cmake_minimum_required(VERSION 3.10)


#
project(VideoPipeline)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmakes)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")


OPTION(ENABLE_CUDA      "使能CUDA库"               ON)
OPTION(ENABLE_TENSORRT  "使能TensorRT推理引擎"      ON)
OPTION(ENABLE_ONNX      "使能ONNXRUNTIME推理引擎"   ON)
OPTION(ENABLE_OPENCV    "包含OPENCV库"              ON)
# OPTION(ENABLE_HWCODEC "使用硬件编解码能力" OFF)

# message
message(STATUS "-----------------------------------")
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "ENABLE_TENSORRT: ${ENABLE_TENSORRT}")
message(STATUS "ENABLE_ONNX: ${ENABLE_ONNX}")
message(STATUS "ENABLE_OPENCV: ${ENABLE_OPENCV}")
message(STATUS "-----------------------------------")


## OpenCV
if(ENABLE_OPENCV)
    find_package(OpenCV REQUIRED)
    
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
endif()


find_package(Threads REQUIRED)
find_package(X11 REQUIRED)


# FFmpeg
find_package(FFmpeg REQUIRED)
# 添加头文件路径与库文件
include_directories(${FFMPEG_INCLUDE_DIRS})
link_directories(${FFMPEG_LIBRARIES})


## ONNX
if(ENABLE_ONNX)
    # ONNX依赖配置
    add_definitions(-DEnable_ONNX)
endif()


## CUDA
if (ENABLE_CUDA)
    # CUDA依赖配置
    add_definitions(-DEnable_CUDA)
    include(${CMAKE_SOURCE_DIR}/cmakes/cuda.cmake)
endif ()

## TensorRT
if (ENABLE_TENSORRT)
    # TensorRT依赖配置
    include(${CMAKE_SOURCE_DIR}/cmakes/tensorrt.cmake)
    add_definitions(-DEnable_TRT)
endif ()


## source

add_subdirectory(src)
# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty)

# 添加可执行文件
add_executable(demo demo/welding.cpp)

# 添加链接库目录
link_directories(${CMAKE_SOURCE_DIR}/libs)
# link_directories(/home/wnk/code/software/onnxruntime-linux-x64-1.17.1/lib)
# target_link_libraries(demo onnxruntime)

target_link_libraries(demo zj_pipe cuda_lib ${CMAKE_SOURCE_DIR}/libs/libonnxruntime.so pthread 
    ${X11_LIBRARIES} ${OpenCV_LIBS} ${TRT_LIBS} ${CUDA_LIBS} ${CUDNN_LIBS} ${FFMPEG_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})