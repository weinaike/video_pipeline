
#ifndef __ZJV_EXTRASDATA_H__
#define __ZJV_EXTRASDATA_H__

#include "BaseData.h"
#include "FrameData.h"
namespace ZJVIDEO
{
    struct Rect
    {
        int x; // left
        int y; // top
        int width;
        int height;
    };

    struct DetectBox
    {
        float x1 = -1;
        float y1 = -1;
        float x2 = -1;
        float y2 = -1;
        float score = -1;       // 置信度
        int label = -1;         // 网络输出标签
        int main_category;      // 标签系统的主类别
        int sub_category;       // 标签系统的子类别
        int track_id = -1;      // 跟踪ID
        int instance_id;
        std::shared_ptr<FrameData> mask;    // 实例分割掩码
    };

    struct DetectBoxCategory
    {
        DetectBox * original_b;
        int label;               // 网络输出标签
        int main_category;       // 标签系统的主类别
        int sub_category;        // 标签系统的子类别
        float score;
    };
    
    // 检测结果，实例分割结果
    class DetectResultData : public BaseData
    {
    public:
        explicit DetectResultData() : BaseData(ZJV_DATATYPE_DETECTRESULT)
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
        explicit ClassifyResultData() : BaseData(ZJV_DATATYPE_CLASSIFYRESULT)
        {
            data_name = "ClassifyResult";
        }
        ~ClassifyResultData() override = default;
        std::vector<DetectBoxCategory> detect_box_categories;

        virtual int append(std::shared_ptr<BaseData>& data_ptr) override;    
    };
    // 语义分割结果
    class SegmentResultData : public BaseData
    {
    public:
        explicit SegmentResultData() : BaseData(ZJV_DATATYPE_SEGMENTRESULT)
        {
            data_name = "SegmentResult";
        }
        ~SegmentResultData() override = default;
        std::shared_ptr<FrameData> mask;
        std::shared_ptr<FrameData> confidence_map;

        virtual int append(std::shared_ptr<BaseData>& data_ptr) override;    
    };


    // 检测结果，分类结果等
    class ExtraData : public BaseData
    {
    public:
        explicit ExtraData() : BaseData(ZJV_DATATYPE_EXTRA)
        {
            data_name = "Extra";
        }
        ~ExtraData() override = default;
    };

    class EventData : public BaseData
    {
    public:
        explicit EventData() : BaseData(ZJV_DATATYPE_EVENT)
        {
            data_name = "Event";
        }
        ~EventData() override = default;
        std::shared_ptr<const FrameData> frame; // 帧数据

    
    };



} // namespace ZJVIDEO

#endif //__ZJV_EXTRASDATA_H__