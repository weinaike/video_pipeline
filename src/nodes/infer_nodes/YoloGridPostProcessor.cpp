
#include "YoloGridPostProcessor.h"

#include "../../logger/easylogging++.h"
#define YoloLOG "YoloGrid"

namespace ZJVIDEO
{
YoloGridPostProcessor::YoloGridPostProcessor()
{
    el::Loggers::getLogger(YoloLOG);
}

int YoloGridPostProcessor::parse_configure(const std::string &cfg_file)
{           
    std::ifstream i(cfg_file);
    if(i.is_open() == false)
    {
        std::cout << "open cfg_file failed";
        return ZJV_STATUS_ERROR;
    }
    nlohmann::json j;
    i >> j;    
    try{
        if (j.contains("postprocess"))
        {
            m_post_type = j["postprocess"]["post_process_type"];
            m_output_data_type = j["postprocess"]["output_data_type"];
            m_num_classes = j["postprocess"]["num_classes"];
            m_conf_thres = static_cast<float>(j["postprocess"]["confidence_threshold"]);
            m_iou_thres = static_cast<float>(j["postprocess"]["nms_threshold"]);
        }
    }
    catch (nlohmann::json::exception& e) {
        CLOG(ERROR, YoloLOG) << "parse postprocess failed" << e.what();
    } 

    assert(m_output_data_type == "DetectResult");
    assert(m_post_type == "YoloGrid");
    // 打印配置信息
    CLOG(INFO, YoloLOG) << "-------------YoloGrid "<<cfg_file<<"----------------";
    CLOG(INFO, YoloLOG) << "post_type:          " << m_post_type;
    CLOG(INFO, YoloLOG) << "output_data_type:   " << m_output_data_type;
    CLOG(INFO, YoloLOG) << "num_classes: [" << m_num_classes << "]";
    CLOG(INFO, YoloLOG) << "conf_thresh: [" << m_conf_thres << "]";
    CLOG(INFO, YoloLOG) << "iou_thresh:  [" << m_iou_thres << "]";
    CLOG(INFO, YoloLOG) << "-----------------------------------------------------";
    
    return ZJV_STATUS_OK;
}

int YoloGridPostProcessor::run(const std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_roi_results)
{
    for(int i = 0; i < outputs.size(); i++)
    {
        if(outputs[i].name_ != "output")
        {
            CLOG(ERROR, YoloLOG) << "output node name not supported now";
            assert(0);
        }      

        const float * output_data = outputs[i].cpu_data();
        std::vector<int> output_shape = outputs[i].shape();
        int bs = output_shape[0];
        int num = output_shape[1];
        int dim = output_shape[2];
        int nc = dim - 5; // 类别数量
        assert(nc == m_num_classes);
        

        for(int j = 0; j < bs; j++)
        {
            std::shared_ptr<DetectResultData> detect_result_data = std::make_shared<DetectResultData>();
            detect_result_data->detect_boxes.clear();

            for(int k = 0; k < num; k++)
            {
                float score = output_data[j*num*dim + k*dim + 4];               
                int max_index = -1;

                float max_obj_conf = 0.0;
                for (int t = 0; t < nc; t++) {
                    auto obj_conf = output_data[j*num*dim + k*dim + 5 + t];;
                    if (obj_conf >= max_obj_conf) {
                        max_obj_conf = obj_conf;
                        max_index = t;
                    }
                }

                float conf = max_obj_conf * score;


                if(conf < m_conf_thres) continue;


                float x1 = output_data[j*num*dim + k*dim] - output_data[j*num*dim + k*dim + 2]/2;
                float y1 = output_data[j*num*dim + k*dim + 1] - output_data[j*num*dim + k*dim + 3]/2;
                float x2 = x1 + output_data[j*num*dim + k*dim + 2];
                float y2 = y1 + output_data[j*num*dim + k*dim + 3];
                DetectBox detect_box;
                detect_box.x1 = x1;
                detect_box.y1 = y1;
                detect_box.x2 = x2;
                detect_box.y2 = y2;
                detect_box.score = conf;
                detect_box.label = max_index;
                detect_result_data->detect_boxes.push_back(detect_box);
            }
            NMS(detect_result_data->detect_boxes, m_iou_thres ) ;
            // 打印结果

            Rect roi = frame_roi_results[j]->roi;

            float scalex  = frame_roi_results[j]->scale_x;
            float scaley  = frame_roi_results[j]->scale_y;
            int padx = frame_roi_results[j]->padx;
            int pady = frame_roi_results[j]->pady;

            for(int k = 0; k < detect_result_data->detect_boxes.size(); k++)
            {
                detect_result_data->detect_boxes[k].x1 = detect_result_data->detect_boxes[k].x1 / scalex + roi.x - padx / scalex;
                detect_result_data->detect_boxes[k].y1 = detect_result_data->detect_boxes[k].y1 / scaley + roi.y - pady / scaley;
                detect_result_data->detect_boxes[k].x2 = detect_result_data->detect_boxes[k].x2 / scalex + roi.x - padx / scalex;
                detect_result_data->detect_boxes[k].y2 = detect_result_data->detect_boxes[k].y2 / scaley + roi.y - pady / scaley;
                // CLOG(INFO, INFER_LOG) << "detect_boxes: " << detect_result_data->detect_boxes[k].x1 << " " << detect_result_data->detect_boxes[k].y1 << " " << detect_result_data->detect_boxes[k].x2 << " " << detect_result_data->detect_boxes[k].y2 << " " << detect_result_data->detect_boxes[k].score << " " << detect_result_data->detect_boxes[k].label;
            }
            frame_roi_results[j]->result.push_back(detect_result_data);
        }
    }

    return ZJV_STATUS_OK;
}

REGISTER_POST_CLASS(YoloGrid)

} // namespace ZJVIDEO
