
#ifndef __ZJV_DEVALLOCATOR_H__
#define __ZJV_DEVALLOCATOR_H__

#include "BaseAllocator.h"


namespace ZJVIDEO {

class DevAllocator : public BaseAllocator {
public:
	void* allocate(size_t bytesCount);
	void deallocate(void* p);
	void operator()(void* p);

	static void Copy(void* dst, void* src, const size_t count);
    static void CopyDevToHost(void* dst, void* src, const size_t count);
    static void CopyHostToDev(void* dst, void* src, const size_t count);
    int m_device_id = 0;
};

}


#endif