#ifndef __ZJV_BASEALLOCATOR_H__
#define __ZJV_BASEALLOCATOR_H__

#include "public/PublicData.h"

namespace ZJVIDEO {

class PUBLIC_API BaseAllocator {
public:
	virtual void *allocate(size_t bytesCount) = 0;
	virtual void deallocate(void* p) = 0;
	virtual void operator()(void* p) = 0;

	static void Copy(void* dst, void* src, const size_t count){}
	unsigned getAlignment()
    {
        return 4U;
    }
};

}




#endif