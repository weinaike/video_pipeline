

#ifndef ZJVIDEO_FRAMEDATA_H
#define ZJVIDEO_FRAMEDATA_H


#include "BaseData.h"
#include <vector>
#include <memory>

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
    explicit FrameData(BaseDataType data_type): BaseData(data_type) {}
    // 析构函数
    ~FrameData() override = default;

    int width;   // 图像宽度
    int height;  // 图像高度
    int channel; // 图像通道数
    int depth;   // 图像深度
    int format;  // 图像格式 ImageFormat
    int fps;     // 帧率

    void *data;  // 图像数据
    int64_t pts; // 时间戳
    int64_t frame_id; // 帧号
    int64_t frame_type; // 帧类型   FrameType

    std::vector<std::shared_ptr<BaseData> >  m_extras;
};

}  // namespace Data


#endif
