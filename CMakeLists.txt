cmake_minimum_required(VERSION 3.10)


#
project(VideoPipeline)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmakes)


# Detect Windows platform
if(WIN32)
    message(STATUS "Configuring for Windows platform")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DWIN32")
    add_compile_options("/Zi" "/Od" "/source-charset:utf-8" "/execution-charset:utf-8" "/w")
    set(CMAKE_C_COMPILER "cl")
    set(CMAKE_CXX_COMPILER "cl")    
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring for Linux platform")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINUX")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "g++")    
endif()



OPTION(ENABLE_CUDA      "使能CUDA库"               OFF)
OPTION(ENABLE_TENSORRT  "使能TensorRT推理引擎"      OFF)
OPTION(ENABLE_ONNX      "使能ONNXRUNTIME推理引擎"   ON)
OPTION(ENABLE_OPENCV    "包含OPENCV库"              OFF)
OPTION(ENABLE_FFMPEG    "包含FFMPEG库"              OFF)
# OPTION(ENABLE_HWCODEC "使用硬件编解码能力" OFF)

# message
message(STATUS "-----------------------------------")
message(STATUS "ENABLE_CUDA: ${ENABLE_CUDA}")
message(STATUS "ENABLE_TENSORRT: ${ENABLE_TENSORRT}")
message(STATUS "ENABLE_ONNX: ${ENABLE_ONNX}")
message(STATUS "ENABLE_OPENCV: ${ENABLE_OPENCV}")
message(STATUS "ENABLE_FFMPEG: ${ENABLE_FFMPEG}")
message(STATUS "-----------------------------------")


## OpenCV
if(ENABLE_OPENCV)

    if(WIN32)
        # Set the OpenCV_DIR to the directory containing OpenCVConfig.cmake for Windows
        set(OpenCV_DIR "C:/path/to/your/opencv/build")  # 修改为你的 OpenCV 安装路径
    elseif(UNIX AND NOT APPLE)
        # Set the OpenCV_DIR to the directory containing OpenCVConfig.cmake for Linux
        # set(OpenCV_DIR "/path/to/your/opencv/build")  # 修改为你的 OpenCV 安装路径
        
    else()
        message(FATAL_ERROR "Unsupported platform for OpenCV")
    endif()

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

    if(WIN32)
        link_directories(${OpenCV_LIB_DIR})
    endif()
endif()


## ONNX
if(ENABLE_ONNX)
    # ONNX依赖配置
    add_definitions(-DEnable_ONNX)

    if(WIN32)
        # Windows specific ONNX configuration
        set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/3rdparty/win_x64/OnnxRuntime)
        # set(ONNX_LIB ${CMAKE_SOURCE_DIR}/3rdparty/win_x64/OnnxRuntime/lib/onnxruntime.lib)
        include_directories(${ONNXRUNTIME_DIR}/include)
        find_library(ONNX_LIB NAMES onnxruntime PATHS ${ONNXRUNTIME_DIR}/lib)
        if(NOT ONNX_LIB)
            message(FATAL_ERROR "ONNX Runtime library not found")
        else()
            message(STATUS "ONNX_LIB: ${ONNX_LIB}")
        endif()

    elseif(UNIX AND NOT APPLE)
        # Linux specific ONNX configuration
        set(ONNX_LIB ${CMAKE_SOURCE_DIR}/libs/libonnxruntime.so)
        include_directories(${CMAKE_SOURCE_DIR}/3rdparty/OnnxRuntime)
        message(STATUS "ONNX_LIB: ${ONNX_LIB}")        
    endif()

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



if (ENABLE_FFMPEG)
    add_definitions(-DEnable_FFMPEG)
    # FFmpeg
    if (WIN32)
        set(FFMPEG_INCLUDE_DIRS "D:/code/camera/OtherLibs/ffmpeg/include")
        link_directories("D:/code/camera/OtherLibs/ffmpeg/lib")
        set(FFMPEG_LIBRARIES "avcodec" "avformat" "avutil" "swscale")
    elseif(UNIX AND NOT APPLE)
        # set(FFMPEG_INCLUDE_DIRS "/usr/local/include" "/usr/include")
        # set(FFMPEG_LIBRARIES "/usr/local/lib" "/usr/lib")
        find_package(FFmpeg REQUIRED)
    else()
        message(FATAL_ERROR "Unsupported platform for FFmpeg")
    endif()

    # 添加头文件路径与库文件
    include_directories(${FFMPEG_INCLUDE_DIRS})
    link_directories(${FFMPEG_LIBRARIES})
endif()

## source

add_subdirectory(src)
# 头文件目录
include_directories(${CMAKE_SOURCE_DIR}/src)

# 添加可执行文件
add_executable(demo demo/welding.cpp)

# 添加链接库目录
link_directories(${CMAKE_SOURCE_DIR}/libs)
# link_directories(/home/wnk/code/software/onnxruntime-linux-x64-1.17.1/lib)
# target_link_libraries(demo onnxruntime)

set(LIBS zj_pipe)

if (ENABLE_CUDA)
    list(APPEND LIBS ${CUDA_LIBS})
    list(APPEND LIBS cuda_lib)
endif()

if (WIN32)
    message(STATUS "Configuring for Windows platform")
elseif(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    find_package(X11 REQUIRED)
    # include_directories(${X11_INCLUDE_DIR})
    list(APPEND LIBS pthread)
    list(APPEND LIBS ${X11_LIBRARIES})
    list(APPEND LIBS ${CMAKE_THREAD_LIBS_INIT})    
endif()

if (ENABLE_ONNX)
    list(APPEND LIBS ${ONNX_LIB})
endif()

if (ENABLE_OPENCV)
    list(APPEND LIBS ${OpenCV_LIBS})
endif()

if (ENABLE_TENSORRT)
    list(APPEND LIBS ${TRT_LIBS})
    list(APPEND LIBS ${CUDNN_LIBS} )    
endif()

if (ENABLE_FFMPEG)
    list(APPEND LIBS ${FFMPEG_LIBRARIES})
    message(STATUS "FFMPEG_LIBRARIES: ${FFMPEG_LIBRARIES}")
endif()


if (WIN32)

    add_custom_command(TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/3rdparty/win_x64/OnnxRuntime/lib/onnxruntime.dll
        $<TARGET_FILE_DIR:demo>)

    add_custom_command(TARGET demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/libs/$<CONFIG>/zj_pipe.dll
        $<TARGET_FILE_DIR:demo>)

endif()

target_link_libraries(demo ${LIBS})
