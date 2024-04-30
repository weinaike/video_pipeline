#include "SyncedMemory.h"
#include <assert.h>
#include <iostream>
namespace ZJVIDEO
{

    SyncedMemory::SyncedMemory(size_t size, void *ptr, SyncedHead head = ZJV_SYNCEHEAD_HEAD_AT_CPU)
    {
        if (head == ZJV_SYNCEHEAD_HEAD_AT_CPU)
        {
            cpu_ptr_ = ptr;
            own_cpu_data_ = true;
            gpu_ptr_ = NULL;
            own_gpu_data_ = false;
            head_ = head;
            size_ = size;
        }
        else if (head == ZJV_SYNCEHEAD_HEAD_AT_GPU)
        {
            cpu_ptr_ = NULL;
            own_cpu_data_ = false;
            gpu_ptr_ = ptr;
            own_gpu_data_ = true;
            head_ = head;
            size_ = size;
        }
        else
        {
            std::cout << "SyncedMemory::SyncedMemory(size_t size, void *ptr, SyncedHead head) error" << std::endl;
            assert(0);
        }
    }

    SyncedMemory::SyncedMemory()
        : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), own_cpu_data_(false), own_gpu_data_(false), head_(ZJV_SYNCEHEAD_UNINITIALIZED)
    {
    }

    SyncedMemory::SyncedMemory(size_t size)
    {
        cpu_ptr_ = malloc_allocator_.allocate(size);
        own_cpu_data_ = true;
        gpu_ptr_ = NULL;
        own_gpu_data_ = false;
        head_ = ZJV_SYNCEHEAD_HEAD_AT_CPU;
        size_ = size;
    }

    SyncedMemory::SyncedMemory(size_t size, void *ptr, void *gpu_ptr)
    {
        assert(ptr != NULL);
        assert(gpu_ptr != NULL);
        cpu_ptr_ = ptr;
        own_cpu_data_ = true;
        gpu_ptr_ = gpu_ptr;
        own_gpu_data_ = true;
        head_ = ZJV_SYNCEHEAD_SYNCED;
        size_ = size;
    }

    SyncedMemory::~SyncedMemory()
    {
        if (cpu_ptr_ && own_cpu_data_)
        {
            malloc_allocator_.deallocate(cpu_ptr_);
            cpu_ptr_ = NULL;
        }

        if (gpu_ptr_ && own_gpu_data_)
        {
            dev_allocator_.deallocate(gpu_ptr_);
            gpu_ptr_ = NULL;
        }
    }

    SyncedMemory::SyncedMemory(const SyncedMemory &other)
    {
        size_ = other.size_;
        head_ = other.head_;
        device_ = other.device_;
        own_cpu_data_ = other.own_cpu_data_;
        own_gpu_data_ = other.own_gpu_data_;

        if (other.own_cpu_data_)
        {
            cpu_ptr_ = malloc_allocator_.allocate(size_);
            malloc_allocator_.Copy(cpu_ptr_, other.cpu_ptr_, size_);
        }

        if (other.own_gpu_data_)
        {
            gpu_ptr_ = dev_allocator_.allocate(size_);
            dev_allocator_.Copy(gpu_ptr_, other.gpu_ptr_, size_);
        }

    }

    inline void SyncedMemory::to_cpu()
    {
        switch (head_)
        {
        case ZJV_SYNCEHEAD_UNINITIALIZED:
            if (own_cpu_data_ == false)
            {
                cpu_ptr_ = malloc_allocator_.allocate(size_);
                own_cpu_data_ = true;                
                head_ = ZJV_SYNCEHEAD_HEAD_AT_CPU;
            }
            break;
        case ZJV_SYNCEHEAD_HEAD_AT_CPU:
            break;
        case ZJV_SYNCEHEAD_HEAD_AT_GPU:
            if (own_cpu_data_ == false)
            {
                cpu_ptr_ = malloc_allocator_.allocate(size_);
                own_cpu_data_ = true;
            }
            dev_allocator_.CopyDevToHost(cpu_ptr_, gpu_ptr_, size_);                
            head_ = ZJV_SYNCEHEAD_SYNCED;            
            break;
        case ZJV_SYNCEHEAD_SYNCED:
            break;
       }        

    }

    inline void SyncedMemory::to_gpu()
    {
        switch (head_)
        {
        case ZJV_SYNCEHEAD_UNINITIALIZED:
            if (own_gpu_data_ == false)
            {
                gpu_ptr_ = dev_allocator_.allocate(size_);
                own_gpu_data_ = true;                
                head_ = ZJV_SYNCEHEAD_HEAD_AT_GPU;
            }
            break;
        case ZJV_SYNCEHEAD_HEAD_AT_CPU:
            if (own_gpu_data_ == false)
            {
                gpu_ptr_ = dev_allocator_.allocate(size_);
                own_gpu_data_ = true;
            }
            dev_allocator_.CopyHostToDev(gpu_ptr_, cpu_ptr_, size_);                
            head_ = ZJV_SYNCEHEAD_SYNCED;
            break;
        case ZJV_SYNCEHEAD_HEAD_AT_GPU:
            break;
        case ZJV_SYNCEHEAD_SYNCED:
            break;
        }        
    }

    const void *SyncedMemory::cpu_data()
    {
        to_cpu();
        return  (const void*)cpu_ptr_;
    }

    const void *SyncedMemory::gpu_data()
    {
        to_gpu();
        return (const void*)gpu_ptr_;
    }

    void *SyncedMemory::mutable_cpu_data()
    {
        to_cpu();
        head_ = ZJV_SYNCEHEAD_HEAD_AT_CPU;
        return cpu_ptr_;
    }

    void *SyncedMemory::mutable_gpu_data()
    {
        to_gpu();
        head_ = ZJV_SYNCEHEAD_HEAD_AT_GPU;
        return gpu_ptr_;
    }
    int SyncedMemory::set_device_id(int device_id)
    {
        
        dev_allocator_.set_device_id(device_id);
        device_ = dev_allocator_.get_device_id();
        return 0;
    }
}
