
#ifndef __ZJV_EXTRASDATA_H__
#define __ZJV_EXTRASDATA_H__

#include "BaseData.h"
#include "FrameData.h"
#include "Shape.h"
#include "Blob.h"
namespace ZJVIDEO
{

    enum TrackStatus
    {
        ZJV_TRACK_STATUS_INIT = 0,
        ZJV_TRACK_STATUS_DETECTED = 1,
        ZJV_TRACK_STATUS_PREDICTED = 2,
        ZJV_TRACK_STATUS_LOST = 3,
    };


    struct DetectBox
    {
        // 基本检测框信息
        float x1;
        float y1;
        float x2;
        float y2;
        float score;       // 置信度
        int label;         // 网络输出标签
        int main_category;      // 标签系统的主类别
        int sub_category;       // 标签系统的子类别
        // 实例分割相关参数
        int instance_id;
        std::shared_ptr<FrameData> mask;    // 实例分割掩码
        // 跟踪相关参数
        int track_id;      // 跟踪ID
        int track_status;  // 跟踪状态  TrackStatus
        std::vector<Rect> track_boxes; // 跟踪框
    };

    struct ObjectAttribute
    {
        DetectBox * original_b;
        int label;              // 网络输出标签
        int attribute;          // 属性 标签
        int attr_sub_category;  // 属性类别 标签
        float attr_value;       // 属性 值
        float score;
    };
    
    // 检测结果，实例分割结果
    class DetectResultData : public BaseData
    {
    public:
        explicit DetectResultData(BaseDataType type = ZJV_DATATYPE_DETECTRESULT) : BaseData(type)
        {
            data_name = "DetectResult";
        }
        ~DetectResultData() override = default;
        std::vector<DetectBox> detect_boxes;

        virtual int append(std::shared_ptr<BaseData>& data_ptr) override;    
    };


    class ClassifyResultData : public BaseData
    {
    public:
        explicit ClassifyResultData(BaseDataType type = ZJV_DATATYPE_CLASSIFYRESULT) : BaseData(type)
        {
            data_name = "ClassifyResult";
        }
        ~ClassifyResultData() override = default;
        std::vector<ObjectAttribute> obj_attr_info;

        virtual int append(std::shared_ptr<BaseData>& data_ptr) override;    
    };
    // 语义分割结果
    class SegmentResultData : public BaseData
    {
    public:
        explicit SegmentResultData(BaseDataType type = ZJV_DATATYPE_SEGMENTRESULT) : BaseData(type)
        {
            data_name = "SegmentResult";
        }
        ~SegmentResultData() override = default;
        std::shared_ptr<FrameData> mask;
        std::shared_ptr<FrameData> confidence_map;

        virtual int append(std::shared_ptr<BaseData>& data_ptr) override;    
    };
    // 图像缓存结果
    class ImageCahceData : public BaseData
    {
    public:
        explicit ImageCahceData(BaseDataType type = ZJV_DATATYPE_IMAGECACHE ) : BaseData(type)
        {
            data_name = "ImageCache";
            images.clear();
        }
        ~ImageCahceData() override = default;
        std::vector<std::shared_ptr<FrameData>> images;
    };


    class FeatureCacheData : public BaseData
    {
    public:
        explicit FeatureCacheData(BaseDataType type = ZJV_DATATYPE_FEATURECACHE ) : BaseData(type)
        {
            data_name = "FeatureCache";
            feature = nullptr;
        }
        ~FeatureCacheData() override = default;
        std::shared_ptr<FBlob> feature;
    };

    // 检测结果，分类结果等
    class ExtraData : public BaseData
    {
    public:
        explicit ExtraData(BaseDataType type = ZJV_DATATYPE_EXTRA) : BaseData(type)
        {
            data_name = "Extra";
        }
        ~ExtraData() override = default;
    };

    class EventData : public BaseData
    {
    public:
        explicit EventData(BaseDataType type = ZJV_DATATYPE_EVENT) : BaseData(type)
        {
            data_name = "Event";
        }
        ~EventData() override = default;
        std::shared_ptr<const FrameData> frame; // 帧数据
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


} // namespace ZJVIDEO

#endif //__ZJV_EXTRASDATA_H__