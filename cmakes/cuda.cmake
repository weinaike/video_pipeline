
# cuda环境的修改路径
#

if(WIN32)
    message(STATUS "Configuring CUDA for Windows platform")
    # set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
    # set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    link_directories(${CUDAToolkit_LIBRARY_DIR})
    message(STATUS "CUDAToolkit_LIBRARY_DIR: ${CUDAToolkit_LIBRARY_DIR}")
    set(CUDAToolkit_LIBRARIES
        ${CUDAToolkit_LIBRARY_DIR}/cuda.lib
        ${CUDAToolkit_LIBRARY_DIR}/cublas.lib
        ${CUDAToolkit_LIBRARY_DIR}/cudart.lib
        # Add other required CUDA libraries here
    )
	set(CMAKE_CUDA_ARCHITECTURES 80) 
    set(CUDA_LIBS ${CUDA_LIBS} ${CUDAToolkit_LIBRARIES})
    message(STATUS "CUDA_LIBS: ${CUDA_LIBS}")
elseif(UNIX AND NOT APPLE)    
    message(STATUS "Configuring CUDA for Linux platform")
    # set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8)
    set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.3)
    set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")   
	find_package(CUDA REQUIRED)

	include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

	set(CUDA_LIBS ${CUDA_LIBS} cuda cublas cudart)

else()
    message(FATAL_ERROR "Unsupported platform for CUDA")
endif()


