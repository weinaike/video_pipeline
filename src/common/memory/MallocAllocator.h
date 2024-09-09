#ifndef __ZJV_MALLOC_ALLOCATOR_H__
#define __ZJV_MALLOC_ALLOCATOR_H__

#include <exception>
#include "BaseAllocator.h"
#include <memory>
#include <cstring>
#include <cstdlib>

namespace	ZJVIDEO {

class PUBLIC_API MallocAllocator : public BaseAllocator {
public:

	void *allocate(size_t bytesCount) {
		void* p = NULL;
		p = malloc(bytesCount);
		
		if (p == NULL) {
			throw std::bad_alloc();
		}

		memset(p,0,bytesCount);
		return p;
	}

	void deallocate(void* p) {
		if(p != NULL)
		{
			free(p);
		}
		
	}

	void operator()(void* p) {
		if(p != NULL)
		{
			free(p);
		}
	}

	static void Copy(void* dst, void* src, const size_t count)
	{
		memcpy(dst, src, count);
	}
};

}

#endif