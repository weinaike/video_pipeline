#ifndef __ZJV_SYNCMEMORY_H__
#define __ZJV_SYNCMEMORY_H__

#include <memory>
#include "MallocAllocator.h"
#include "DevAllocator.h"
namespace ZJVIDEO
{

enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };


class SyncedMemory
{
public:
    SyncedMemory() = delete;
    explicit SyncedMemory(size_t size);
    explicit SyncedMemory(size_t size, void *ptr, SyncedHead head);
    explicit SyncedMemory(size_t size, void *ptr, void* gpu_ptr) ;   
    // 拷贝构造函数
    SyncedMemory(const SyncedMemory &other);
    ~SyncedMemory();
    const void *cpu_data();
    const void *gpu_data();
    void *mutable_cpu_data();
    void *mutable_gpu_data();
    size_t size() const { return size_; }
    

private:
    void to_cpu();
    void to_gpu();
    void *cpu_ptr_;
    void *gpu_ptr_;
    size_t size_;
    SyncedHead head_;
    bool own_cpu_data_;
    bool own_gpu_data_;
    int device_;

    MallocAllocator malloc_allocator_;
    DevAllocator dev_allocator_;

};

} // namespace ZJVIDEO

#endif // __ZJV_BASEMEMORY_H__