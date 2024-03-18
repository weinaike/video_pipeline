

#ifndef ZJVIDEO_FRAMEDATA_H
#define ZJVIDEO_FRAMEDATA_H


#include "BaseData.h"
#include <vector>
#include <memory>
#include "memory/SyncedMemory.h"
namespace ZJVIDEO {

enum FrameType {
    ZJV_FRAMETYPE_UNKNOWN = 0,
    ZJV_FRAMETYPE_VIDEO,  // 视频帧
    ZJV_FRAMETYPE_IMAGE,  // 图片帧
    ZJV_FRAMETYPE_MAX
};

enum ImageFormat{
    ZJV_IMAGEFORMAT_UNKNOWN = 0,
    ZJV_IMAGEFORMAT_GRAY8,  // GRAY8
    ZJV_IMAGEFORMAT_GRAY10LE,  // GRAY10LE
    ZJV_IMAGEFORMAT_GRAY12LE,  // GRAY12LE
    ZJV_IMAGEFORMAT_GRAY16LE,  // GRAY16LE
    ZJV_IMAGEFORMAT_RGB24,  // RGB24
    ZJV_IMAGEFORMAT_BGR24,  // BGR24
    ZJV_IMAGEFORMAT_PRGB24,  // PRGB24
    ZJV_IMAGEFORMAT_YUV420P,  // YUV420P
    ZJV_IMAGEFORMAT_YUV422P,  // YUV422P
    ZJV_IMAGEFORMAT_YUV444P,  // YUV444P
    ZJV_IMAGEFORMAT_NV21,  // NV21
    ZJV_IMAGEFORMAT_NV12,  // NV12

    ZJV_IMAGEFORMAT_MAX
};

// 用于存储帧数据的类
class FrameData : public BaseData {
public:
    explicit FrameData(): BaseData(ZJV_DATATYPE_FRAME) 
    {
        data_name = "Frame";
        width = 0;
        stride = 0;
        height = 0;
        channel = 0;
        depth = 0;
        format = ZJV_IMAGEFORMAT_UNKNOWN;
        fps = 0;
        pts = 0;
        camera_id = 0;
        frame_id = 0;
        frame_type = ZJV_FRAMETYPE_UNKNOWN;
        data = nullptr;
    }
    // 析构函数
    ~FrameData() override = default;
    // 拷贝构造函数
    FrameData(const FrameData &other): BaseData(ZJV_DATATYPE_FRAME)
    {
        width = other.width;
        stride = other.stride;
        height = other.height;
        channel = other.channel;
        depth = other.depth;
        format = other.format;
        fps = other.fps;
        
        pts = other.pts;
        camera_id = other.camera_id;
        frame_id = other.frame_id;
        frame_type = other.frame_type;

        // 深度拷贝内存
        data = std::make_shared<SyncedMemory>(*(other.data.get()));

        data_name = other.data_name;
    }

    int width;   // 图像宽度
    int stride;  // 图像步长
    int height;  // 图像高度
    int channel; // 图像通道数
    int depth;   // 图像深度
    int format;  // 图像格式 ImageFormat
    int fps;     // 帧率

    std::shared_ptr<SyncedMemory> data;  // 图像数据
    int64_t pts;            // 时间戳
    int camera_id;          // 相机ID
    int64_t frame_id;       // 帧号
    int64_t frame_type;     // 帧类型   FrameType
};

}  // namespace ZJVIDEO

#endif
