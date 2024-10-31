#ifndef __PUBLICDATA_H__
#define __PUBLICDATA_H__

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32) || defined(_WIN64) || defined(WIN32) || defined(WIN64)
    #define PUBLIC_API __declspec(dllexport)
    // #ifdef BUILDING_DLL
    //     #define PUBLIC_API __declspec(dllexport)
    // #else
    //     #define PUBLIC_API __declspec(dllimport)
    // #endif
#else
    #define PUBLIC_API
#endif




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
class PUBLIC_API BaseData {
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



class PUBLIC_API EventData : public BaseData
{
public:
    explicit EventData(BaseDataType type = ZJV_DATATYPE_EVENT) : BaseData(type) { data_name = "Event"; }
    ~EventData() override = default;
    std::vector<std::shared_ptr<const BaseData> >  extras; // 额外数据
};


class PUBLIC_API WeldResultData : public BaseData
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

}

#endif // __PUBLICDATA_H__