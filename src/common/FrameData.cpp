#include "FrameData.h"
#include "Function.h"
namespace ZJVIDEO
{
    FrameData::FrameData(): BaseData(ZJV_DATATYPE_FRAME) 
    {
        data_name = "Frame";
        width = 0;
        stride = 0;
        height = 0;
        depth = 0;
        format = ZJV_IMAGEFORMAT_UNKNOWN;

        fps = 0;
        pts = 0;
        camera_id = 0;
        frame_id = 0;

        data = nullptr;
    }

    static unsigned int get_depth(int fmt )
    {
        int depth = 0;
        if(fmt == ZJV_IMAGEFORMAT_GRAY8) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY10LE) depth = 10;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY12LE) depth = 12;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY16LE) depth = 16;
        else if(fmt == ZJV_IMAGEFORMAT_RGB24) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_BGR24) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_RGBP) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_YUV420P) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_YUV422P) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_YUV444P) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_NV21) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_NV12) depth = 8;
        else if(fmt == ZJV_IMAGEFORMAT_FLOAT32) depth = 32;
        else depth = 0;
        return depth;
    }

    static int get_stride(int fmt, int width, bool align = false)
    {
        int stride = 0;
        unsigned int depth = get_depth(fmt);
        if(align)
        {
            if(fmt == ZJV_IMAGEFORMAT_RGB24 || fmt == ZJV_IMAGEFORMAT_BGR24)
            {
                stride = uSnapUp(uDivUp(depth, 8u) * width * 3, 4u);
            }
            else
            {
                stride = uSnapUp(uDivUp(depth, 8u) * width, 4u) ;
            }
        }
        else
        {
            if(fmt == ZJV_IMAGEFORMAT_RGB24 || fmt == ZJV_IMAGEFORMAT_BGR24)
            {
                stride = uDivUp(depth, 8u) * width * 3;
            }
            else
            {
                stride = uDivUp(depth, 8u) * width;
            }
        }

        return stride;
    }

    int FrameData::channel () const
    {
        int fmt = format;
        int channel = 0;
        if(fmt == ZJV_IMAGEFORMAT_GRAY8) channel = 1;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY10LE) channel = 1;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY12LE) channel = 1;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY16LE) channel = 1;
        else if(fmt == ZJV_IMAGEFORMAT_RGB24) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_BGR24) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_RGBP) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_YUV420P) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_YUV422P) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_YUV444P) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_NV21) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_NV12) channel = 3;
        else if(fmt == ZJV_IMAGEFORMAT_FLOAT32) channel = 1;
        else channel = 0;
        if(channel == 0) std::cout <<__FILE__ << __LINE__ << " Unknown format: " << fmt;
        return channel;
    }

    static int  malloc_data(std::shared_ptr<SyncedMemory> & data, int fmt, int w, int h, bool align = false)
    {
        int stride = get_stride(fmt, w, align);
        int size = 0;

        if(fmt == ZJV_IMAGEFORMAT_GRAY8) size =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY10LE)  size  =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY12LE)  size  =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_GRAY16LE)  size  =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_RGB24) size  =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_BGR24) size  =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_RGBP) size  =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_FLOAT32) size  =  stride * h ;
        else if(fmt == ZJV_IMAGEFORMAT_YUV420P) size = stride * h * 3 / 2;
        else if(fmt == ZJV_IMAGEFORMAT_YUV422P) size = stride * h * 2;
        else if(fmt == ZJV_IMAGEFORMAT_YUV444P) size = stride * h * 3;
        else if(fmt == ZJV_IMAGEFORMAT_NV21) size = stride * h * 3 / 2;
        else if(fmt == ZJV_IMAGEFORMAT_NV12) size = stride * h * 3 / 2;
        else
        {
            std::cout << __FILE__ << __LINE__ << " yuv data is not support yet: " << fmt;
            return -1;
        }

        data = std::make_shared<SyncedMemory>(size);
        return size;
    }


    FrameData::FrameData(int w, int h, int fmt, bool align):
        BaseData(ZJV_DATATYPE_FRAME) 
    {
        data_name = "Frame"; 
        alignment = align;

        width = w;
        height = h;
        format = fmt;
        depth = get_depth(fmt);
        stride = get_stride(fmt, w, align);
        int ret = malloc_data(data, fmt, w, h, align);
        if(ret < 0)
        {
            data.reset();
        }

        
        fps = 0;
        pts = 0;
        camera_id = 0;
        frame_id = 0;

    }

    FrameData::FrameData(const FrameData &other): BaseData(ZJV_DATATYPE_FRAME)
    {
        width = other.width;
        stride = other.stride;
        height = other.height;
        depth = other.depth;
        format = other.format;


        fps = other.fps;        
        pts = other.pts;
        camera_id = other.camera_id;
        frame_id = other.frame_id;


        // 深度拷贝内存
        data = std::make_shared<SyncedMemory>(*(other.data.get()));

        data_name = other.data_name;
    }

    REGISTER_DATA_CLASS(Frame)





    // REGISTER_DATA_CLASS(Video)

}