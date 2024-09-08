
# cuda环境的修改路径
#

if(WIN32)
    message(STATUS "Configuring CUDA for Windows platform")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
elseif(UNIX AND NOT APPLE)    
    message(STATUS "Configuring CUDA for Linux platform")
    # set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.3)
    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")    
else()
    message(FATAL_ERROR "Unsupported platform for CUDA")
endif()

find_package(CUDA REQUIRED)

include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

set(CUDA_LIBS ${CUDA_LIBS} cuda cublas cudart)



# 把所有的cu文件都编译成一个动态库
file(GLOB_RECURSE CU_SRC ${CMAKE_SOURCE_DIR}/src/cuda_kernels/*.cu)

cuda_add_library(cuda_lib SHARED ${CU_SRC})