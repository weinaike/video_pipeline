message("配置cuda环境")

#
# cuda环境的修改路径
#
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)

find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CUDA_LIBS ${CUDA_LIBS} cuda cublas cudart)
