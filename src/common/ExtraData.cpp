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


}