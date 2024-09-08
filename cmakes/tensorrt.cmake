
# 1. 配置tensorrt环境

if(WIN32)
    message(STATUS "Configuring TensorRT and cuDNN for Windows platform")
    set(TENSORRT_ROOT_DIR "C:/TensorRT-8.6.1.6_cuda12.0")
    set(CUNDD_ROOT_DIR "C:/cudnn-cuda12")
elseif(UNIX AND NOT APPLE)
    message(STATUS "Configuring TensorRT and cuDNN for Linux platform")
    set(TENSORRT_ROOT_DIR "/usr/local/TensorRT-8.6.1.6_cuda12.0")
    # set(CUNDD_ROOT_DIR /usr/local/cudnn-cuda11/)
    set(CUNDD_ROOT_DIR "/usr/local/cudnn-cuda12")
else()
    message(FATAL_ERROR "Unsupported platform for TensorRT and cuDNN")
endif()

include_directories(${TENSORRT_ROOT_DIR}/include)
link_directories(${TENSORRT_ROOT_DIR}/lib)

set(TRT_LIBS nvinfer nvonnxparser  ${TENSORRT_ROOT_DIR}/lib/libnvinfer_plugin.so.8)

# 2. 配置cuDNN环境
include_directories(${CUNDD_ROOT_DIR}/include)
link_directories(${CUNDD_ROOT_DIR}/lib)

set(CUDNN_LIBS cudnn)
