
# 1. 配置tensorrt环境
#
# set(TENSORRT_ROOT_DIR /usr/local/TensorRT-8.6.1.6_cuda11.8/)
set(TENSORRT_ROOT_DIR /usr/local/TensorRT-8.6.1.6_cuda12.0/)

include_directories(${TENSORRT_ROOT_DIR}/include)
link_directories(${TENSORRT_ROOT_DIR}/lib)

set(TRT_LIBS nvinfer nvonnxparser  ${TENSORRT_ROOT_DIR}/lib/libnvinfer_plugin.so.8)


# set(CUNDD_ROOT_DIR /usr/local/cudnn-cuda11/)
set(CUNDD_ROOT_DIR /usr/local/cudnn-cuda12/)

include_directories(${CUNDD_ROOT_DIR}/include)
link_directories(${CUNDD_ROOT_DIR}/lib)

set(CUDNN_LIBS cudnn)
