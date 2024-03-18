
#ifndef __ZJV_EXTRASDATA_H__
#define __ZJV_EXTRASDATA_H__

#include "BaseData.h"

namespace ZJVIDEO
{

class FrameData;

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
        float score = -1;
        int label = -1;
        int track_id = -1;
    };

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