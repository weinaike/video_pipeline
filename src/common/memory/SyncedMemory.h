#ifndef __ZJV_SYNCMEMORY_H__
#define __ZJV_SYNCMEMORY_H__

#include <memory>
#include "MallocAllocator.h"
#include "DevAllocator.h"
namespace ZJVIDEO
{

    enum SyncedHead
    {
        ZJV_SYNCEHEAD_UNINITIALIZED,
        ZJV_SYNCEHEAD_HEAD_AT_CPU,
        ZJV_SYNCEHEAD_HEAD_AT_GPU,
        ZJV_SYNCEHEAD_SYNCED
    };

    class PUBLIC_API SyncedMemory
    {
    public:
        SyncedMemory();
        explicit SyncedMemory(size_t size);
        explicit SyncedMemory(size_t size, void *ptr, SyncedHead head);
        explicit SyncedMemory(size_t size, void *ptr, void *gpu_ptr);
        // 拷贝构造函数
        SyncedMemory(const SyncedMemory &other);
        // SyncedMemory& operator=(const SyncedMemory &other) = delete;
        ~SyncedMemory();
        const void *cpu_data();
        const void *gpu_data();
        void *mutable_cpu_data();
        void *mutable_gpu_data();
        size_t size() const { return size_; }
        int set_device_id(int device_id);
        int get_device_id() const { return device_; }

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