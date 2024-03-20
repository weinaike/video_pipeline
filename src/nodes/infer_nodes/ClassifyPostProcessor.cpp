
#include "ClassifyPostProcessor.h"

#include "../../logger/easylogging++.h"
#define clsLOG "Classify"

namespace ZJVIDEO
{
ClassifyPostProcessor::ClassifyPostProcessor()
{
    el::Loggers::getLogger(clsLOG);
    m_post_type = "Classify";
}

int ClassifyPostProcessor::parse_json(const nlohmann::json & j)
{           
    
    m_input_names = j["input_names"].get<std::vector<std::string>>();
    m_output_data_type = j["output_data_type"];
    m_num_classes = j["num_classes"];
    m_main_categories = j["main_category"].get<std::vector<int>>();
    m_sub_categories = j["sub_category"].get<std::vector<int>>();
    assert(m_output_data_type == "ClassifyResult");
    // 打印配置信息
    CLOG(INFO, clsLOG) << "-------------YoloGrid ----------------";
    CLOG(INFO, clsLOG) << "post_type:          " << m_post_type;
    for (size_t i = 0; i < m_input_names.size(); i++)
    {
        CLOG(INFO, clsLOG) << "input_names:        " << m_input_names[i];
    }
    
    
    CLOG(INFO, clsLOG) << "output_data_type:   " << m_output_data_type;
    CLOG(INFO, clsLOG) << "num_classes: [" << m_num_classes << "]";
    CLOG(INFO, clsLOG) << "---------------------------------------";
    
    return ZJV_STATUS_OK;
}

static void softmax(float* output, int num) {
    float max_output = *std::max_element(output, output + num);
    float sum = 0.0f;

    for (int i = 0; i < num; i++) {
        output[i] = std::exp(output[i] - max_output);
        sum += output[i];
    }

    for (int i = 0; i < num; i++) {
        output[i] /= sum;
    }
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
        int bs = output_shape[0];
        int num = output_shape[1];
        assert(num == m_num_classes);


        float * output_data = (float *)outputs[i].mutable_cpu_data();
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
            
            DetectBoxCategory box_cls;
            box_cls.label = max_index;
            box_cls.score = max_score;
            if(m_main_categories.size() > 0) box_cls.main_category = m_main_categories[max_index];
            else box_cls.main_category = 0;


            if(m_sub_categories.size() > 0) box_cls.sub_category = m_sub_categories[max_index];
            else box_cls.sub_category = 0;

            cls_result->detect_box_categories.push_back(box_cls);
 
            frame_roi_results[j]->result.push_back(cls_result);
        }
    }

    return ZJV_STATUS_OK;
}

REGISTER_POST_CLASS(Classify)

} // namespace ZJVIDEO
