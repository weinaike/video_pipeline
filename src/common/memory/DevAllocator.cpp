
#include <exception>
#include <cstdio>
#include "DevAllocator.h"

namespace ZJVIDEO {
#ifdef USE_CUDA

#include <cuda_runtime.h>
void* DevAllocator::allocate(size_t bytesCount) {
	void* p = NULL;
	if (cudaMalloc((void**)&p, bytesCount) != cudaSuccess) {
		throw std::bad_alloc();
	}
	return p;
}

void DevAllocator::deallocate(void* p) {
	if (cudaFree(p) != cudaSuccess) {
		throw std::bad_alloc();
	}
}

void DevAllocator::operator()(void* p) {
	if (cudaFree(p) != cudaSuccess) {
		throw std::bad_alloc();
	}
}

void DevAllocator::Copy(void* dst, void* src, const size_t count) {
	cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "Memory copy error: %s\n", cudaGetErrorString(error));
	}
}

void DevAllocator::CopyDevToHost(void* dst, void* src, const size_t count)
{
	cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess) {
		fprintf(stderr, "Memory copy error: %s\n", cudaGetErrorString(error));
	}
}
void DevAllocator::CopyHostToDev(void* dst, void* src, const size_t count)
{
	cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "Memory copy error: %s\n", cudaGetErrorString(error));
	}
}

#else

void* DevAllocator::allocate(size_t bytesCount) {
	return NULL;
}

void DevAllocator::deallocate(void* p) {
}

void DevAllocator::operator()(void* p) {
}

void DevAllocator::Copy(void* dst, void* src, const size_t count) {
}

void DevAllocator::CopyDevToHost(void* dst, void* src, const size_t count)
{

}
void DevAllocator::CopyHostToDev(void* dst, void* src, const size_t count)
{

}

#endif


}