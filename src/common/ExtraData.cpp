#include "ExtraData.h"
#include "StatusCode.h"


namespace ZJVIDEO
{


    int DetectResultData::append(std::shared_ptr<BaseData>& data_ptr)
    {
        const std::shared_ptr<DetectResultData> data = std::dynamic_pointer_cast<DetectResultData>(data_ptr);
        if(data!= nullptr)
        {
            detect_boxes.insert(detect_boxes.end(), data->detect_boxes.begin(), data->detect_boxes.end());
        }

        return ZJV_STATUS_OK;
    }
    REGISTER_DATA_CLASS(DetectResult)


    int ClassifyResultData::append(std::shared_ptr<BaseData>& data_ptr)
    {
        const std::shared_ptr<ClassifyResultData> data = std::dynamic_pointer_cast<ClassifyResultData>(data_ptr);
        if(data!= nullptr)
        {
            detect_box_categories.insert(detect_box_categories.end(), data->detect_box_categories.begin(), data->detect_box_categories.end());
        }

        return ZJV_STATUS_OK;
    }
    REGISTER_DATA_CLASS(ClassifyResult)


    int SegmentResultData::append(std::shared_ptr<BaseData>& data_ptr)
    {
        const std::shared_ptr<SegmentResultData> data = std::dynamic_pointer_cast<SegmentResultData>(data_ptr);
        if(data!= nullptr)
        {
            // Traverse all elements, compare the pixel values of confidence_map, keep the maximum value,
            // update confidence_map, and synchronize update the pixel values in mask
            if(data->mask == nullptr || data->confidence_map == nullptr)
            {
                std::cout<<"SegmentResultData::append: mask or confidence_map is nullptr"<<std::endl;
                return ZJV_STATUS_ERROR;
            }
            if(mask == nullptr || confidence_map == nullptr)
            {
                mask = data->mask;
                confidence_map = data->confidence_map;
                return ZJV_STATUS_OK;
            }

            if((data->mask->width != mask->width) || (data->mask->height != mask->height)) 
            {
                std::cout<<"SegmentResultData::append: mask size not match"<<" "<< data->mask->width
                <<" " << mask->width << " " << data->mask->height <<" " << mask->height <<std::endl;
                return ZJV_STATUS_ERROR;
            }
            int w = mask->width;
            int h = mask->height;

            for(int i = 0; i < h; i++)
            {
                const float * in_ptr = (float *) data->confidence_map->data->cpu_data();
                const unsigned char * in_mask_ptr = (unsigned char *) data->mask->data->cpu_data();
                for(int j = 0; j < w; j++)
                {
                    float * ptr = (float *) confidence_map->data->cpu_data();
                    unsigned char * mask_ptr = (unsigned char * ) mask->data->cpu_data();
                    if(ptr[i * h + w] < in_ptr[i * h + w])
                    {
                        ptr[i * h + w] = in_ptr[i * h + w];
                        mask_ptr[i * h + w] = in_mask_ptr[i * h + w];
                    }
                }
            }
        }

        return ZJV_STATUS_OK;
    }
    REGISTER_DATA_CLASS(SegmentResult)



}   // namespace ZJVIDEO