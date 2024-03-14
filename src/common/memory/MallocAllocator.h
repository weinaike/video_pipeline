#ifndef __ZJV_MALLOC_ALLOCATOR_H__
#define __ZJV_MALLOC_ALLOCATOR_H__

#include <exception>
#include "BaseAllocator.h"
#include <memory>


namespace	ZJVIDEO {

class MallocAllocator : public BaseAllocator {
public:

	void *allocate(size_t bytesCount) {
		void* p = NULL;
		p = malloc(bytesCount);
		memset(p,0,bytesCount);
		if (p == NULL) {
			throw std::bad_alloc();
		}
		return p;
	}

	void deallocate(void* p) {
		free(p);
	}

	void operator()(void* p) {
		free(p);
	}

	static void Copy(void* dst, void* src, const size_t count)
	{
		memcpy(dst, src, count);
	}
};

}

#endif