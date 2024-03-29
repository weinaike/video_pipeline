
#include <exception>
#include <cstdio>
#include "DevAllocator.h"

#include <iostream>
// #define Enable_CUDA

#ifdef Enable_CUDA
#include <new>
#include <cuda_runtime.h>
#endif

namespace ZJVIDEO
{

#ifdef Enable_CUDA

	void *DevAllocator::allocate(size_t bytesCount)
	{
		void *p = NULL;
		if (cudaMalloc((void **)&p, bytesCount) != cudaSuccess)
		{
			std::cout<<__FILE__<<" cudaMalloc fail "<<__LINE__<<std::endl;
			throw std::bad_alloc();
		}
		cudaMemset(p, 0, bytesCount);
		return p;
	}

	void DevAllocator::deallocate(void *p)
	{
		if (p != NULL)
		{
			if (cudaFree(p) != cudaSuccess)
			{
				std::cout<<__FILE__<<" cudaFree fail "<<__LINE__<<std::endl;
				throw std::bad_alloc();
			}
		}
	}

	void DevAllocator::operator()(void *p)
	{
		if (p != NULL)
		{
			if (cudaFree(p) != cudaSuccess)
			{
				std::cout<<__FILE__<< " cudaFree fail "<<__LINE__<<std::endl;
				throw std::bad_alloc();
			}
		}
	}

	void DevAllocator::Copy(void *dst, void *src, const size_t count)
	{
		cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Memory copy error: %s\n", cudaGetErrorString(error));
		}
	}

	void DevAllocator::CopyDevToHost(void *dst, void *src, const size_t count)
	{
		cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Memory copy error: %s\n", cudaGetErrorString(error));
		}
	}
	void DevAllocator::CopyHostToDev(void *dst, void *src, const size_t count)
	{
		cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Memory copy error: %s\n", cudaGetErrorString(error));
		}
	}
	int DevAllocator::set_device_id(int device_id)
	{
		cudaSetDevice(device_id);
		cudaGetDevice(&m_device_id);
		return 0;
	}

#else

	void *DevAllocator::allocate(size_t bytesCount)
	{
		return NULL;
	}

	void DevAllocator::deallocate(void *p)
	{
	}

	void DevAllocator::operator()(void *p)
	{
	}

	void DevAllocator::Copy(void *dst, void *src, const size_t count)
	{
		std::cout << "CUDA is not enabled" << std::endl;
	}

	void DevAllocator::CopyDevToHost(void *dst, void *src, const size_t count)
	{
		std::cout << "CUDA is not enabled" << std::endl;
	}
	void DevAllocator::CopyHostToDev(void *dst, void *src, const size_t count)
	{
		std::cout << "CUDA is not enabled" << std::endl;
	}
	int DevAllocator::set_device_id(int device_id)
	{
		std::cout << "CUDA is not enabled" << std::endl;
		return 0;
	}

#endif

}