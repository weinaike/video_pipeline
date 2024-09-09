
#ifndef __ZJV_DEVALLOCATOR_H__
#define __ZJV_DEVALLOCATOR_H__

#include "BaseAllocator.h"


namespace ZJVIDEO {

class PUBLIC_API DevAllocator : public BaseAllocator {
public:
	void* allocate(size_t bytesCount);
	void deallocate(void* p);
	void operator()(void* p);

	static void Copy(void* dst, void* src, const size_t count);
    static void CopyDevToHost(void* dst, void* src, const size_t count);
    static void CopyHostToDev(void* dst, void* src, const size_t count);
	int set_device_id(int device_id);
	int get_device_id() const { return m_device_id; }
    int m_device_id = 0;
};

}


#endif