
#include "ClassifyPostProcessor.h"

#include "logger/easylogging++.h"
#define clsLOG "Classify"

namespace ZJVIDEO
{
ClassifyPostProcessor::ClassifyPostProcessor()
{
    el::Loggers::getLogger(clsLOG);
    m_post_type = "Classify";
    m_input_names.clear();
    m_output_data_type = "";
    m_main_categories.clear();
    m_sub_categories.clear();
    
    // private
    m_num_classes = 0;
    m_algorithm = ZJV_CLASSIFY_ALGORITHM_SOFTMAX;
    m_attr_value_norm = 1.0f;

}

int ClassifyPostProcessor::parse_json(const nlohmann::json & j)
{           
    try{
        m_input_names = j["input_names"].get<std::vector<std::string>>();
        m_output_data_type = j["output_data_type"];
        assert(m_output_data_type == "ClassifyResult");


        if(j.contains("main_attribute"))
            m_main_categories = j["main_attribute"].get<std::vector<int>>();
        if(j.contains("sub_attribute"))
            m_sub_categories = j["sub_attribute"].get<std::vector<int>>();
        
        if(j.contains("attr_value_norm_param"))
        {
            m_attr_value_norm = j["attr_value_norm_param"].get<float>();
        }
        if(j.contains("algorithm"))
        {
            std::string algorithm = j["algorithm"];
            if(algorithm == "softmax")
            {
                m_algorithm = ZJV_CLASSIFY_ALGORITHM_SOFTMAX;
            }
            else if(algorithm == "sigmoid")
            {
                m_algorithm = ZJV_CLASSIFY_ALGORITHM_SIGMOID;
            }
            else if(algorithm == "mse")
            {
                m_algorithm = ZJV_CLASSIFY_ALGORITHM_MSE;
            }
            else
            {
                m_algorithm = ZJV_CLASSIFY_ALGORITHM_SOFTMAX;
            }
        }
        

        // private
        m_num_classes = j["num_classes"];
    }
    catch (nlohmann::json::exception& e) {
        CLOG(ERROR, clsLOG) << "parse preprocess failed" << e.what();
    }
    // 打印配置信息
    CLOG(INFO, clsLOG) << "-------------Classify ----------------";
    CLOG(INFO, clsLOG) << "post_type:          " << m_post_type;
    for (size_t i = 0; i < m_input_names.size(); i++)
    {
        CLOG(INFO, clsLOG) << "input_names:        " << m_input_names[i];
    }
    CLOG(INFO, clsLOG) << "output_data_type:   " << m_output_data_type;
    CLOG(INFO, clsLOG) << "num_classes: [" << m_num_classes << "]";
    CLOG(INFO, clsLOG) << "algorithm:          " << m_algorithm;
    CLOG(INFO, clsLOG) << "main_attribute:     " << m_main_categories.size();
    CLOG(INFO, clsLOG) << "sub_attribute:      " << m_sub_categories.size();
    CLOG(INFO, clsLOG) << "attr_value_norm:    " << m_attr_value_norm;
    CLOG(INFO, clsLOG) << "---------------------------------------";
    
    return ZJV_STATUS_OK;
}


int ClassifyPostProcessor::run( std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_roi_results)
{
    for(int i = 0; i < outputs.size(); i++)
    {
        if (std::find(m_input_names.begin(), m_input_names.end(), outputs[i].name_) == m_input_names.end()) 
        {
            continue;
        }

        std::vector<int> output_shape = outputs[i].shape();
        assert(output_shape.size() == 2);
        int bs = output_shape[0];
        int num = output_shape[1];
        assert(num == m_num_classes);


        float * output_data = (float *)outputs[i].mutable_cpu_data();

        if(m_algorithm == ZJV_CLASSIFY_ALGORITHM_SOFTMAX)
        {
            // softmax
            softmax(output_data + i * num, num);

            for(int j = 0; j < bs; j++)
            {
                // 获取该维度下置信度最高的值及其标签
                float max_score = 0.0f;
                int max_index = 0;
                for(int k = 0; k < num; k++)
                {
                    if(output_data[j * num + k] > max_score)
                    {
                        max_score = output_data[j * num + k];
                        max_index = k;
                    }
                }
                
                std::shared_ptr<ClassifyResultData> cls_result = std::make_shared<ClassifyResultData>();
                
                ObjectAttribute box_cls = {0};
                box_cls.label = max_index;
                box_cls.score = max_score;
                if(m_main_categories.size() > 0) box_cls.attribute = m_main_categories[max_index];
                else box_cls.attribute = 0;


                if(m_sub_categories.size() > 0) box_cls.attr_sub_category = m_sub_categories[max_index];
                else box_cls.attr_sub_category = 0;

                cls_result->obj_attr_info.push_back(box_cls);
    
                frame_roi_results[j]->result.push_back(cls_result);
            }
        }
        else if (m_algorithm == ZJV_CLASSIFY_ALGORITHM_MSE)
        {
            for(int j = 0; j < bs; j++)
            {
                // 获取该维度下置信度最高的值及其标签
                std::shared_ptr<ClassifyResultData> cls_result = std::make_shared<ClassifyResultData>();
                for(int k = 0; k < num; k++)
                {
                    ObjectAttribute box_cls = {0};
                    box_cls.label = k;
                    box_cls.attr_value = output_data[j * num + k] * m_attr_value_norm;
                    if(m_main_categories.size() > 0) box_cls.attribute = m_main_categories[k];
                    else box_cls.attribute = 0;

                    cls_result->obj_attr_info.push_back(box_cls);   
                }   
                frame_roi_results[j]->result.push_back(cls_result);           
            }
        }        
    }

    return ZJV_STATUS_OK;
}

REGISTER_POST_CLASS(Classify)

} // namespace ZJVIDEO
