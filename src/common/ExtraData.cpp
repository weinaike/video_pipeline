#include "ExtraData.h"

namespace ZJVIDEO
{


    int DetectResultData::append(std::shared_ptr<BaseData>& data_ptr)
    {
        const std::shared_ptr<DetectResultData> data = std::dynamic_pointer_cast<DetectResultData>(data_ptr);
        if(data!= nullptr)
        {
            detect_boxes.insert(detect_boxes.end(), data->detect_boxes.begin(), data->detect_boxes.end());
        }

        return 0;
    }
    REGISTER_DATA_CLASS(DetectResult)


    int ClassifyResultData::append(std::shared_ptr<BaseData>& data_ptr)
    {
        const std::shared_ptr<ClassifyResultData> data = std::dynamic_pointer_cast<ClassifyResultData>(data_ptr);
        if(data!= nullptr)
        {
            detect_box_categories.insert(detect_box_categories.end(), data->detect_box_categories.begin(), data->detect_box_categories.end());
        }

        return 0;
    }
    REGISTER_DATA_CLASS(ClassifyResult)



}