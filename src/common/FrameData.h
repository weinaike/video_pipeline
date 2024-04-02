

#ifndef ZJVIDEO_FRAMEDATA_H
#define ZJVIDEO_FRAMEDATA_H

#include "BaseData.h"
#include <vector>
#include <memory>
#include "memory/SyncedMemory.h"
namespace ZJVIDEO
{

    // enum FrameType {
    //     ZJV_FRAMETYPE_UNKNOWN = 0,
    //     ZJV_FRAMETYPE_VIDEO,  // 视频帧
    //     ZJV_FRAMETYPE_IMAGE,  // 图片帧
    //     ZJV_FRAMETYPE_MAX
    // };

    enum ImageFormat
    {
        ZJV_IMAGEFORMAT_UNKNOWN = 0,
        ZJV_IMAGEFORMAT_GRAY8,    // GRAY8
        ZJV_IMAGEFORMAT_GRAY10LE, // GRAY10LE
        ZJV_IMAGEFORMAT_GRAY12LE, // GRAY12LE
        ZJV_IMAGEFORMAT_GRAY16LE, // GRAY16LE
        ZJV_IMAGEFORMAT_RGB24,    // RGB24
        ZJV_IMAGEFORMAT_BGR24,    // BGR24
        ZJV_IMAGEFORMAT_RGBP,     // RGBP
        ZJV_IMAGEFORMAT_YUV420P,  // YUV420P
        ZJV_IMAGEFORMAT_YUV422P,  // YUV422P
        ZJV_IMAGEFORMAT_YUV444P,  // YUV444P
        ZJV_IMAGEFORMAT_NV21,     // NV21
        ZJV_IMAGEFORMAT_NV12,     // NV12
        // 以上是常用的图像格式，以下是不常用的图像格式

        ZJV_IMAGEFORMAT_FLOAT32, // FLOAT32

        ZJV_IMAGEFORMAT_MAX
    };

    // enum PixelDtype
    // {
    //     ZJV_PIXEL_DTYPE_UNKNOWN = 0, // "Unknown"
    //     ZJV_PIXEL_DTYPE_FLOAT32 = 1, // "float32"
    //     ZJV_PIXEL_DTYPE_UINT16 = 2, // "float32"
    //     ZJV_PIXEL_DTYPE_UINT8 = 3,   // "uint8"
    // };

    // 用于存储帧数据的类
    class FrameData : public BaseData
    {
    public:
        explicit FrameData();

        explicit FrameData(int w, int h, int fmt = ZJV_IMAGEFORMAT_RGBP, bool align = true);

        // 析构函数
        ~FrameData() override = default;
        // 拷贝构造函数
        FrameData(const FrameData &other);

        int channel() const; // 获取图像通道数
        // 图像相关参数
        int width;  // 图像宽度
        int stride; // 图像步长 uSnapUp(uSnapUp(depth, 8) * width, 4) 向上 4字节对齐取整
        int height; // 图像高度
        int depth;  // 图像深度 depth bit数
        int format; // 图像格式 ImageFormat


        std::shared_ptr<SyncedMemory> data; // 图像数据
        // 视频相关参数
        int fps;          // 帧率
        int camera_id;    // 相机ID
        int64_t frame_id; // 帧号
        int64_t pts;      // 时间戳

        bool alignment = false; // 是否需要内存4字节对齐（扩展宽）
    };

    // 检测结果，实例分割结果
    class VideoData : public BaseData
    {
    public:
        explicit VideoData(std::string url, int id = 0) : BaseData(ZJV_DATATYPE_VIDEO)
        {
            video_path = url;
            camera_id = id;
            data_name = "Video";
        }
        ~VideoData() override = default;

        std::string video_path;
        int camera_id;
    };

} // namespace ZJVIDEO

#endif
