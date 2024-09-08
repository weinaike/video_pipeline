#ifndef __PUBLICDATA_H__
#define __PUBLICDATA_H__

#include <chrono>
#include <memory>
#include <string>

namespace ZJVIDEO
{
    
enum BaseDataType {
    ZJV_DATATYPE_UNKNOWN = 0,
    ZJV_DATATYPE_VIDEO,    // 视频数据
    ZJV_DATATYPE_FRAME,    // 帧数据
    ZJV_DATATYPE_CONTROL,  // 控制数据
    ZJV_DATATYPE_CONFIG,   // 配置数据
    ZJV_DATATYPE_EXTRA,    // 额外数据，检测结果，识别结果等
    ZJV_DATATYPE_FLOW,     // 流数据
    ZJV_DATATYPE_DETECTRESULT,    //检测结果，实例分割结果
    ZJV_DATATYPE_DETECTRESULT_TRACK,    //跟踪结果
    ZJV_DATATYPE_CLASSIFYRESULT,  //分类结果
    ZJV_DATATYPE_SEGMENTRESULT,   //语义分割结果
    ZJV_DATATYPE_IMAGECACHE,   //图像缓存
    ZJV_DATATYPE_FEATURECACHE, //特征缓存


    ZJV_DATATYPE_EVENT = 1000,    // 事件数据
    ZJV_DATATYPE_EVENT_WELDING,    // 焊接事件数据

    ZJV_DATATYPE_MAX
};



// 用于存储数据的基类
class BaseData {
public:
    BaseData() = delete;
    explicit BaseData(BaseDataType data_type) : data_type(data_type)  { create_time = std::chrono::system_clock::now(); }

    virtual ~BaseData() = default;
    virtual int append(std::shared_ptr<BaseData>& data) { return 0; }

    BaseDataType get_data_type() const  { return data_type; }

    BaseDataType                          data_type;    // 数据类型
    std::chrono::system_clock::time_point create_time;  // 数据创建时间
    std::string                           data_name;    // 数据名称/数据来

};



class EventData : public BaseData
{
public:
    explicit EventData(BaseDataType type = ZJV_DATATYPE_EVENT) : BaseData(type) { data_name = "Event"; }
    ~EventData() override = default;
    std::vector<std::shared_ptr<const BaseData> >  extras; // 额外数据
};


class WeldResultData : public BaseData
{
public:
    explicit WeldResultData(BaseDataType type = ZJV_DATATYPE_EVENT_WELDING) : BaseData(type)
    {
        data_name = "WeldResult";   
        is_enable = false;
        camera_id = 0;
        frame_id = 0;
        weld_status = 0;
        status_score = 0.0f;
        weld_depth = 0.0f;
        front_quality = 0.0f;
        back_quality = 0.0f;

    }
    ~WeldResultData() override = default;
    bool    is_enable;
    int     camera_id;
    int     frame_id;
    int     weld_status;
    float   status_score;
    float   weld_depth;
    float   front_quality;
    float   back_quality; 
};



// class ImageData : public BaseData
// {
// public:
//     explicit ImageData(int w, int h, int fmt = ZJV_IMAGEFORMAT_RGBP, bool align = true);

//     // 析构函数
//     ~FrameData() override = default;

//     int channel() const; // 获取图像通道数
//     // 图像相关参数
//     int width;  // 图像宽度
//     int stride; // 图像步长 uSnapUp(uSnapUp(depth, 8) * width, 4) 向上 4字节对齐取整
//     int height; // 图像高度
//     int depth;  // 图像深度 depth bit数
//     int format; // 图像格式 ImageFormat

//     std::shared_ptr<SyncedMemory> data = nullptr; // 图像数据
//     // 视频相关参数
//     float fps;          // 帧率
//     int camera_id;    // 相机ID
//     int64_t frame_id; // 帧号
//     int64_t pts;      // 时间戳

//     bool alignment = false; // 是否需要内存4字节对齐（扩展宽）
// };




}

#endif // __PUBLICDATA_H__