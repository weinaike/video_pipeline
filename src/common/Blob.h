#ifndef __ZJV_BLOB_H__
#define __ZJV_BLOB_H__

#include "memory/SyncedMemory.h"
#include <vector>
#include <assert.h>

namespace ZJVIDEO
{

    template <typename Dtype>
    class Blob
    {
    public:
        Blob() : data_(NULL), count_(0), capacity_(0) {shape_.clear();}
        explicit Blob(const std::vector<int> &shape)
        {
            shape_ = shape;
            count_ = 1;
            for (int i = 0; i < shape.size(); i++)
            {
                count_ *= shape[i];
            }
            capacity_ = count_;
            data_ = std::make_shared<SyncedMemory>(capacity_ * sizeof(Dtype));
        }
        inline const Dtype *cpu_data() const
        {
            return (const Dtype*)data_->cpu_data();
        }
        inline const Dtype *gpu_data() const
        {
            return (const Dtype*)data_->gpu_data();
        }

        inline Dtype *mutable_cpu_data()
        {
            assert(data_!=nullptr);
            return (Dtype*)data_->cpu_data();
        }

        inline Dtype *mutable_gpu_data()
        {
            assert(data_!=nullptr);
            return (Dtype*)data_->gpu_data();
        }
       
        inline const std::shared_ptr<SyncedMemory> &data() const
        {
            return data_;
        }

        inline int count() const { return count_; }

        inline const std::vector<int> &shape() const { return shape_; }
        std::string name_;

    protected:
        std::shared_ptr<SyncedMemory> data_;
        std::vector<int> shape_;
        int count_;
        int capacity_;
    };

    template class Blob<float>;
    template class Blob<unsigned char>;
    //重命名
    typedef Blob<float> FBlob;
    typedef Blob<unsigned char> U8Blob;
} // namespace ZJVIDEO

#endif