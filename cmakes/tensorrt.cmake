
# 1. 配置tensorrt环境

if(WIN32)
    message(STATUS "Configuring TensorRT and cuDNN for Windows platform")
    set(TENSORRT_ROOT_DIR "D:/code/TensorRT-8.6.1.6")
    set(CUNDD_ROOT_DIR "D:/code/cudnn-windows-x86_64-8.9.7.29_cuda12-archive")
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
if(WIN32)
    set(TRT_LIBS
        ${TENSORRT_ROOT_DIR}/lib/nvinfer.lib
        ${TENSORRT_ROOT_DIR}/lib/nvonnxparser.lib
        ${TENSORRT_ROOT_DIR}/lib/nvinfer_plugin.lib
    )

elseif(UNIX AND NOT APPLE)
    set(TRT_LIBS nvinfer nvonnxparser nvinfer_plugin)
endif()
message(STATUS "TRT_LIBS: ${TRT_LIBS}")
# 2. 配置cuDNN环境
include_directories(${CUNDD_ROOT_DIR}/include)
link_directories(${CUNDD_ROOT_DIR}/lib/x64)

set(CUDNN_LIBS cudnn)
