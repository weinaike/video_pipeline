

#include "SegmentPostProcessor.h"

#include "logger/easylogging++.h"
#define SegLog "SegPost"

#include "CImg/CImg.h"

namespace ZJVIDEO
{

    SegmentPostProcessor::SegmentPostProcessor()
    {
        el::Loggers::getLogger(SegLog);
        m_post_type = "Segment";
        m_input_names.clear();
        m_output_data_type = "";
        m_main_categories.clear();
        m_sub_categories.clear();

        // private
        m_num_classes = 0;
        m_conf_thres = 0.0f;

    }

    int SegmentPostProcessor::parse_json(const nlohmann::json & j)
    {
        m_input_names = j["input_names"].get<std::vector<std::string>>();
        m_output_data_type = j["output_data_type"];
        m_main_categories = j["main_category"].get<std::vector<int>>();
        m_sub_categories = j["sub_category"].get<std::vector<int>>();
        assert(m_output_data_type == "SegmentResult");


        // private
        m_num_classes = j["num_classes"];
        m_conf_thres = j["confidence_threshold"];
        
        // 打印配置信息
        CLOG(INFO, SegLog) << "-------------Segment ----------------";
        CLOG(INFO, SegLog) << "post_type:          " << m_post_type;
        for (size_t i = 0; i < m_input_names.size(); i++)
        {
            CLOG(INFO, SegLog) << "input_names:        " << m_input_names[i];
        }        
        CLOG(INFO, SegLog) << "output_data_type:   " << m_output_data_type;

        // private
        CLOG(INFO, SegLog) << "num_classes: [" << m_num_classes << "]";
        CLOG(INFO, SegLog) << "conf_thresh: [" << m_conf_thres << "]";
        CLOG(INFO, SegLog) << "---------------------------------------";
        return ZJV_STATUS_OK;
    }

    int SegmentPostProcessor::run(std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois)
    {

        for(int i = 0; i < outputs.size(); i++)
        {
            if (std::find(m_input_names.begin(), m_input_names.end(), outputs[i].name_) == m_input_names.end()) 
            {
                continue;
            }

            std::vector<int> output_shape = outputs[i].shape();
            assert(output_shape.size() == 4);
            
            int bs = output_shape[0];
            int num = output_shape[1];
            assert(num == m_num_classes);

            int height = output_shape[2];
            int width = output_shape[3];


            float * output_data = (float *)outputs[i].mutable_cpu_data();
            // softmax

            for(int j = 0; j < bs; j++)
            {
                
                cil::CImg<unsigned char> mask(width, height, 1, 1, 0);
                cil::CImg<float> conf_map(width, height, 1, 1, 0);

                unsigned char * mask_data = (unsigned char *)mask.data();
                float * conf_map_data = (float *)conf_map.data();
                for(int y = 0 ; y < height ; y ++ )
                {
                    for(int x = 0; x < width; x++)
                    {
                        std::vector<float *> confs;
                        for(int k = 0; k < num ; k++)
                        {
                            int idx = j * num * height * width + k * height * width + y * width + x;
                            confs.push_back(&output_data[j]);
                        }
                        softmax(confs);                      

                        float max_score = m_conf_thres;
                        int max_index = 0;
                        for(int k = 0; k < num; k++)
                        {
                            int idx = j * num * height * width + k * height * width + y * width + x;
                            if(output_data[idx] > max_score)
                            {                                
                                max_score = output_data[idx];
                                max_index = k;
                            }
                        }

                        mask_data[y * width + x] = max_index;
                        conf_map_data[y * width + x] = max_score;


                    }
                }
                

                std::shared_ptr<FrameROI> frame_roi = frame_rois[j];
                // output ==> input
                int net_w = frame_roi->input_width;
                int net_h = frame_roi->input_height;
                int net_padx = frame_roi->padx;
                int net_pady = frame_roi->pady;
                float scalex = frame_roi->scale_x;
                float scaley = frame_roi->scale_y;

                mask.resize(net_w, net_h, 1, 1, 1);
                conf_map.resize(net_w, net_h, 1, 1, 1); 

                // input ==> roi
                Rect roi = frame_roi->roi;
                 
                if(frame_roi->resize_type == ZJV_PREPROCESS_RESIZE_LETTERBOX)
                {
                    
                    mask.crop(net_padx, net_pady, net_w - net_padx - 1, net_h - net_pady - 1);
                    conf_map.crop(net_padx, net_pady, net_w - net_padx - 1, net_h - net_pady - 1);
                }
                else if(frame_roi->resize_type == ZJV_PREPROCESS_RESIZE_FILL)
                {
                    // padx, pady
                    cil::CImg<unsigned char> new_mask(net_w - 2 * net_padx , net_h - 2 * net_pady, 1, 1, 0);
                    new_mask.draw_image( 0 - net_padx, 0 - net_pady, mask);
                    mask = new_mask;

                    cil::CImg<float> new_map(net_w - 2 * net_padx, net_h - 2 * net_pady, 1, 1, 0);
                    new_map.draw_image(0 - net_padx, 0 - net_pady, mask);
                    conf_map = new_map;
                }
                
                #if 0
                cil::CImgDisplay disp(mask,"My Image");
                while (!disp.is_closed()) {
                    disp.wait();
                    if (disp.is_key()) {
                        std::cout << "Key pressed: " << disp.key() << std::endl;
                    }
                }
                #endif
                mask.resize(roi.width, roi.height, 1, 1, 1);
                conf_map.resize(roi.width, roi.height, 1, 1, 1);
                // roi ==> original
                
                cil::CImg<unsigned char> original_mask(frame_roi->frame->width, frame_roi->frame->height, 1, 1, 0);
                original_mask.draw_image(roi.x, roi.y, mask);
                cil::CImg<float> original_map(frame_roi->frame->width, frame_roi->frame->height, 1, 1, 0);
                original_map.draw_image(roi.x, roi.y, conf_map);


                std::shared_ptr<FrameData> frame_mask = std::make_shared<FrameData>(frame_roi->frame->width, frame_roi->frame->height, ZJV_IMAGEFORMAT_GRAY8);
                memcpy(frame_mask->data->mutable_cpu_data(), original_mask.data(), original_mask.size() * sizeof(unsigned char));

                std::shared_ptr<FrameData> frame_conf = std::make_shared<FrameData>(frame_roi->frame->width, frame_roi->frame->height,ZJV_IMAGEFORMAT_FLOAT32);
                memcpy(frame_conf->data->mutable_cpu_data(), original_map.data(), original_mask.size() * sizeof(float));

                std::shared_ptr<SegmentResultData> seg_result = std::make_shared<SegmentResultData>();
                seg_result->mask = frame_mask;
                seg_result->confidence_map = frame_conf;
            
                frame_rois[j]->result.push_back(seg_result);
            }
        }

        return ZJV_STATUS_OK;
    }


REGISTER_POST_CLASS(Segment)

} // namespace ZJV