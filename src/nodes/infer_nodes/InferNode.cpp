

#include "InferNode.h"
#include "nlohmann/json.hpp"
namespace ZJVIDEO {

#define INFER_LOG "INFER"

InferNode::InferNode(const NodeParam & param) : BaseNode(param)
{
    el::Loggers::getLogger(INFER_LOG);
    
    parse_configure(param.m_cfg_file);
    init();

    m_max_batch_size = 8; // 根据模型配置设置

    CLOG(INFO, INFER_LOG) << "InferNode::InferNode";
}

InferNode::~InferNode()
{
    CLOG(INFO, INFER_LOG) << "InferNode::~InferNode";
}

int InferNode::parse_configure(std::string cfg_file)
{
    std::ifstream i(cfg_file);
    nlohmann::json j;
    i >> j;
    // 1. 解析出 EngineParameter
    if (j.contains("model") && j.contains("infer") ) 
    {
        m_engine_param.m_model_name = j["model"]["model_name"];
        
        // 字符串转换为枚举类型
        std::string device = j["model"]["device"];
        if(device == "CPU") m_engine_param.m_device = CPU;
        else if(device == "GPU") m_engine_param.m_device = GPU;
        else m_engine_param.m_device = CPU;

        m_engine_param.m_dynamic = j["model"]["dynamic_batch"];
        m_engine_param.m_encrypt = j["model"]["encrypt"];
        m_engine_param.m_engine_type = j["model"]["backend"];   // 这个最重要
        m_engine_param.m_model_path =  j["model"]["model_file"];
        m_engine_param.m_param_path = j["model"]["weight_file"];
        m_engine_param.m_numThread = j["model"]["num_thread"];

        m_engine_param.m_input_node_name = j["infer"]["num_thread"];

        m_engine_param.m_input_node_name = j["infer"]["input_names"].get<std::vector<std::string>>();
        m_engine_param.m_output_node_name = j["infer"]["output_names"].get<std::vector<std::string>>();
        std::vector<std::vector<int>> input_dims = j["infer"]["input_dims"].get<std::vector<std::vector<int>>>();
        assert(m_engine_param.m_input_node_name.size() == input_dims.size());
        for (int i = 0; i < m_engine_param.m_input_node_name.size(); i++)
        {
            m_engine_param.m_input_nodes[m_engine_param.m_input_node_name[i]] = input_dims[i];
        }
    }

    // 2. 解析出 preprocess



    // 3. 解析出 postprocess

    return ZJV_STATUS_OK;
}

int InferNode::init()
{
    // 1. create engine
    m_engine = EngineRegister::CreateEngine(m_engine_param);
    if (m_engine == nullptr)
    {
        CLOG(ERROR, INFER_LOG) << "Create Engine Failed";
        return ZJV_STATUS_ERROR;
    }


    return ZJV_STATUS_OK;
}

int InferNode::process_batch( const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                                std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch)
{

    std::vector<FrameROI> frame_rois;
    prepare(in_metas_batch, frame_rois);

    std::vector<std::shared_ptr<BaseData>> frame_roi_results;
    for(int i = 0; i < frame_rois.size(); i += m_max_batch_size)
    {
        std::vector<FrameROI> batch_frame_rois;

        for(int j = 0; j < m_max_batch_size; j++)
        {
            if((i+j) >= frame_rois.size()) break;
            batch_frame_rois.push_back(frame_rois[i+j]);
        }
        std::vector<std::shared_ptr<BlobData>> inputs;
        preprocess(batch_frame_rois, inputs);
        std::vector<std::shared_ptr<BlobData>> outputs;
        infer(inputs, outputs);
        std::vector<std::shared_ptr<BaseData>> part_results;
        postprocess(outputs, part_results);
        for(int j = 0; j < part_results.size(); j++)
        {
            frame_roi_results.push_back(part_results[j]);
        }
    }

    summary(frame_roi_results, out_metas_batch);

    
    return ZJV_STATUS_OK;
}

int InferNode::prepare( const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                            std::vector<FrameROI> &frame_rois)
{
    if(in_metas_batch.size() == 0) return ZJV_STATUS_ERROR;

    for(int i = 0; i < in_metas_batch.size(); i++)
    {
        
        if(in_metas_batch[i].size() == 1)
        {
            if(in_metas_batch[i][0]->data_name == "Frame")
            {
                std::shared_ptr<const FrameData> frame_data = std::dynamic_pointer_cast<const FrameData>(in_metas_batch[i][0]);
                FrameROI frame_roi;
                frame_roi.frame = frame_data;
                frame_roi.input_vector_id = i;
                frame_roi.roi.x = 0;
                frame_roi.roi.y = 0;
                frame_roi.roi.width = frame_data->width;
                frame_roi.roi.height = frame_data->height;
                frame_rois.push_back(frame_roi);
            }
            else
            {
                CLOG(ERROR, INFER_LOG) << "in_metas_batch without Frame";
                assert(0);
            }
        }
        else
        {   
            std::shared_ptr<const FrameData> frame_data;
            FrameROI frame_roi;
            std::vector<Rect>rois;
            for(int j = 0; j < in_metas_batch[i].size(); j++)
            {
                
                if(in_metas_batch[i][j]->data_name == "Frame")
                {
                    frame_data = std::dynamic_pointer_cast<const FrameData>(in_metas_batch[i][j]);                  
                }
                else if(in_metas_batch[i][j]->data_name == "DetectResult")
                {
                    std::shared_ptr<const DetectResultData> roi_data = std::dynamic_pointer_cast<const DetectResultData>(in_metas_batch[i][j]);
                    for(int k = 0; k < roi_data->detect_boxes.size(); k++)
                    {
                        Rect roi = {0};
                        roi.x = roi_data->detect_boxes[k].left;
                        roi.y = roi_data->detect_boxes[k].top;
                        roi.width = roi_data->detect_boxes[k].right - roi_data->detect_boxes[k].left;
                        roi.height = roi_data->detect_boxes[k].bottom - roi_data->detect_boxes[k].top;
                        rois.push_back(roi);
                    }
                }
                else
                {
                    CLOG(ERROR, INFER_LOG) << "in_metas_batch without Frame or ROI";
                    assert(0);
                }
            }
            for(int j = 0; j < rois.size(); j++) 
            {
                frame_roi.frame = frame_data;
                frame_roi.roi = rois[j];
                frame_roi.input_vector_id = i;
                frame_rois.push_back(frame_roi);
            }
        }
    }
    return ZJV_STATUS_OK;
}

int InferNode::preprocess(const std::vector<FrameROI> &frame_rois, std::vector<std::shared_ptr<BlobData>> inputs)
{
    return ZJV_STATUS_OK;
}
int InferNode::infer(std::vector<std::shared_ptr<BlobData>> inputs, std::vector<std::shared_ptr<BlobData>> outputs)
{
    return ZJV_STATUS_OK;

}
int InferNode::postprocess(std::vector<std::shared_ptr<BlobData>> outputs,std::vector<std::shared_ptr<BaseData>>& frame_roi_results)
{
    return ZJV_STATUS_OK;
}

int InferNode::summary(const std::vector<std::shared_ptr<BaseData>>& frame_roi_results, 
                    std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch)
{
    return ZJV_STATUS_OK;
}


REGISTER_NODE_CLASS(Infer)

} // namespace ZJVIDEO
