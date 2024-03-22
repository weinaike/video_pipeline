

#include "InferNode.h"


namespace ZJVIDEO {

#define INFER_LOG "InferNode"

InferNode::InferNode(const NodeParam & param) : BaseNode(param)
{
    el::Loggers::getLogger(INFER_LOG);
    
    parse_configure(param.m_cfg_file);
    init();
    if(m_engine_param.m_max_batch_size > 8)
    {
        m_max_batch_size = m_engine_param.m_max_batch_size; // 根据模型配置设置
    }
    

    CLOG(INFO, INFER_LOG) << "InferNode::InferNode";
}

InferNode::~InferNode()
{
    CLOG(INFO, INFER_LOG) << "InferNode::~InferNode";
}

int InferNode::parse_configure(std::string cfg_file)
{
    std::ifstream i(cfg_file);
    if(i.is_open() == false)
    {
        CLOG(ERROR, INFER_LOG) << "open cfg_file failed";
        return ZJV_STATUS_ERROR;
    }
    nlohmann::json j;
    i >> j;
    // 1. 解析出 EngineParameter
    if (j.contains("model") && j.contains("infer") ) 
    {
        try {
            m_engine_param.m_model_name = j["model"]["model_name"];
            
            // 字符串转换为枚举类型
            std::string device = j["model"]["device"];
            if(device == "CPU") m_engine_param.m_device = ZJV_DEVICE_CPU;
            else if(device == "GPU") m_engine_param.m_device = ZJV_DEVICE_GPU;
            else m_engine_param.m_device = ZJV_DEVICE_CPU;

            m_engine_param.m_dynamic = j["model"]["dynamic_batch"];
            m_engine_param.m_encrypt = j["model"]["encrypt"];
            m_engine_param.m_engine_type = j["model"]["backend"];   // 这个最重要
            m_engine_param.m_model_path =  j["model"]["model_file"];
            m_engine_param.m_param_path = j["model"]["weight_file"];
            m_engine_param.m_max_batch_size = j["model"]["max_batch_size"];
        }
        catch (nlohmann::json::exception& e) {
            CLOG(ERROR, INFER_LOG) << "parse model failed" << e.what();
        }
        try {
            m_engine_param.m_input_node_name = j["infer"]["input_names"].get<std::vector<std::string>>();
        }
        catch (nlohmann::json::exception& e) {
            CLOG(ERROR, INFER_LOG) << "'input_names' is not an array"  << e.what();
        }
        
        try {
            m_engine_param.m_output_node_name = j["infer"]["output_names"].get<std::vector<std::string>>();
        }
        catch (nlohmann::json::exception& e)  {
            CLOG(ERROR, INFER_LOG) << "'output_names' is not an array"<< e.what();
        }

        try {
            std::vector<std::vector<int>> input_dims = j["infer"]["input_dims"].get<std::vector<std::vector<int>>>();
            assert(m_engine_param.m_input_node_name.size() == input_dims.size());
            for (int i = 0; i < m_engine_param.m_input_node_name.size(); i++)
            {
                m_engine_param.m_input_nodes[m_engine_param.m_input_node_name[i]] = input_dims[i];

                if(input_dims[i].size() < 4 || input_dims[i].size() > 5)
                {
                    CLOG(ERROR, INFER_LOG) << "input_dims size is not supported now, only support 4,5 dims now.";
                    assert(0);
                }
            }
        }catch (nlohmann::json::exception& e) {
            CLOG(ERROR, INFER_LOG) << "'input_dims' is not an array"<< e.what();
        }
    }
    
    // 2. 解析出 preprocess
    int lib_type = ZJV_PREPROCESS_LIB_CIMG;
    if(m_engine_param.m_device == ZJV_DEVICE_CPU)
    {
        lib_type = ZJV_PREPROCESS_LIB_CIMG;
    }
    else if(m_engine_param.m_device == ZJV_DEVICE_GPU)
    {
        lib_type = ZJV_PREPROCESS_LIB_CUDA;
    }
    else
    {
        CLOG(ERROR, INFER_LOG) << "device not supported now";
    }

    if(j.contains("preprocess"))
    {
        if(j["preprocess"].contains("ImageType"))
        {
            nlohmann::json image_types = j["preprocess"]["ImageType"];
            for(auto & image_type: image_types)
            {
                std::shared_ptr<PreProcessor> img_preproc = std::make_shared<PreProcessor>(lib_type);
                img_preproc->parse_json(image_type);
                PreProcessParameter param = img_preproc->get_param();
                m_img_preprocs.push_back(img_preproc);
                m_img_preproc_params.push_back(param);
            }
        }
        
        if(j["preprocess"].contains("VideoType"))
        {
            nlohmann::json video_types = j["preprocess"]["VideoType"];
            if(video_types.size() > 0)
            {
                CLOG(ERROR, INFER_LOG) << "video_type not supported now";
            }
        }
        if(j["preprocess"].contains("FeatureType"))
        {
            nlohmann::json feature_types = j["preprocess"]["FeatureType"];
            if(feature_types.size() > 0)
            {
                CLOG(ERROR, INFER_LOG) << "feature_type not supported now";
            }
        }

        
    } 
    else
    {
        CLOG(ERROR, INFER_LOG) << "preprocess not found";
        assert(0);
    }


    // 3. 解析出 postprocess

    if (j.contains("postprocess"))
    {
        for (nlohmann::json::iterator it = j["postprocess"].begin(); it != j["postprocess"].end(); ++it) 
        {
            std::shared_ptr<PostProcessor> postproc = PostRegister::CreatePost(it.key());
            postproc->parse_json(it.value());
            m_postprocess.push_back(postproc);
        }
    } 
    else
    {
        CLOG(ERROR, INFER_LOG) << "postprocess not found";
        assert(0);
    }   

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
    for(int i = 0; i < in_metas_batch.size(); i++)
    {
        std::vector<std::shared_ptr<BaseData>> out_metas;
        out_metas_batch.push_back(out_metas);
    }
    std::vector<std::shared_ptr<FrameROI>> frame_rois;
    prepare(in_metas_batch, frame_rois);

    for(int i = 0; i < frame_rois.size(); i += m_max_batch_size)
    {
        std::vector<std::shared_ptr<FrameROI>> batch_frame_rois;

        for(int j = 0; j < m_max_batch_size; j++)
        {
            if((i+j) >= frame_rois.size()) break;
            batch_frame_rois.push_back(frame_rois[i+j]);
        }
        std::vector<FBlob> inputs;
        preprocess(batch_frame_rois, inputs);
        std::vector<FBlob> outputs;
        infer(inputs, outputs);
        postprocess(outputs, batch_frame_rois);
    }

    summary(frame_rois, out_metas_batch);
    return ZJV_STATUS_OK;
}

int InferNode::prepare( const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                            std::vector<std::shared_ptr<FrameROI>>  &frame_rois)
{
    if(in_metas_batch.size() == 0) return ZJV_STATUS_ERROR;

    for(int i = 0; i < in_metas_batch.size(); i++)
    {
        
        if(in_metas_batch[i].size() == 1)
        {
            if(in_metas_batch[i][0]->data_name == "Frame")
            {
                std::shared_ptr<const FrameData> frame_data = std::dynamic_pointer_cast<const FrameData>(in_metas_batch[i][0]);
                std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
                frame_roi->frame = frame_data;
                frame_roi->input_vector_id = i;
                frame_roi->roi.x = 0;
                frame_roi->roi.y = 0;
                frame_roi->roi.width = frame_data->width/2*2 ;
                frame_roi->roi.height = frame_data->height/2* 2 ;
                frame_roi->original = nullptr;
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
            std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
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
                        Rect roi;
                        roi.x = roi_data->detect_boxes[k].x1;
                        roi.y = roi_data->detect_boxes[k].y1;
                        roi.width = (roi_data->detect_boxes[k].x2 - roi_data->detect_boxes[k].x1)/2*2;
                        roi.height = (roi_data->detect_boxes[k].y2 - roi_data->detect_boxes[k].y1)/2*2;

                        frame_roi->frame = frame_data;
                        frame_roi->roi = roi;
                        frame_roi->input_vector_id = i;
                        frame_roi->original = &(roi_data->detect_boxes[k]);
                        frame_rois.push_back(frame_roi);                        
                    }
                }
                else
                {
                    CLOG(ERROR, INFER_LOG) << "in_metas_batch without Frame or ROI";
                    assert(0);
                }
            }
            
        }
    }
    return ZJV_STATUS_OK;
}

int InferNode::preprocess(std::vector<std::shared_ptr<FrameROI>>  &frame_rois, std::vector<FBlob> & inputs)
{
    for(int i = 0; i < m_img_preprocs.size(); i++)
    {
        std::vector<int> resize_dims = m_img_preproc_params[i].output_dims;
        resize_dims[0] = frame_rois.size();
        FBlob input_blob(resize_dims);
        input_blob.name_ = m_img_preproc_params[i].output_name;
        m_img_preprocs[i]->run(frame_rois, input_blob, m_img_preproc_params[i]);
        inputs.push_back(input_blob);
    }

    return ZJV_STATUS_OK;
}
int InferNode::infer(std::vector<FBlob> & inputs, std::vector<FBlob> & outputs)
{   
    std::vector<void*> ins;
    std::vector<std::vector<int>> input_shape;
    for(int i = 0; i < inputs.size(); i++)
    {
        ins.push_back(inputs[i].mutable_cpu_data());
        input_shape.push_back(inputs[i].shape());
    }    
    std::vector<std::vector<float>> outs;
    std::vector<std::vector<int>> outputs_shape;

    m_engine->forward(ins, input_shape, outs, outputs_shape); 

    for(int i = 0; i < outs.size(); i++)
    {
        FBlob output_blob(outputs_shape[i]);
        output_blob.name_ = m_engine_param.m_output_node_name[i];
        float * output_data = output_blob.mutable_cpu_data();
        std::memcpy(output_data, outs[i].data(), outs[i].size() * sizeof(float));
        outputs.push_back(output_blob);
    }

    return ZJV_STATUS_OK;

}

int InferNode::postprocess( std::vector<FBlob> & outputs, std::vector<std::shared_ptr<FrameROI>> & frame_roi_results)
{
    for(int i = 0; i < m_postprocess.size(); i++)
    {
        m_postprocess[i]->run(outputs, frame_roi_results);
    }

    return ZJV_STATUS_OK;
}

int InferNode::summary(const std::vector<std::shared_ptr<FrameROI>>  &frame_rois, 
                        std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch)
{
    for(int i = 0; i < out_metas_batch.size(); i++)        
    {
        auto & out_metas = out_metas_batch[i];
        for(auto & output_name: m_nodeparam.m_output_datas)
        {
            std::shared_ptr<BaseData> data = DataRegister::CreateData(output_name);
            // 合并同一帧，相同目标类型的结果，如果有必要，进行nms
            for(auto & frame_roi_result:frame_rois)
            {
                if(frame_roi_result->input_vector_id != i) continue;

                for(auto & result: frame_roi_result->result)
                {
                    if(result->data_name == output_name)
                    {
                        data->append(result);
                    }
                }
            }
            
            if(data->data_name == "DetectResult")
            {
                float nms_thresh = 0.2;
                std::shared_ptr<DetectResultData> detect_result_data_all = std::dynamic_pointer_cast<DetectResultData>(data);
                NMS(detect_result_data_all->detect_boxes,nms_thresh);  
            }
          
            out_metas.push_back(data);
        }
    }
    return ZJV_STATUS_OK;
}


REGISTER_NODE_CLASS(Infer)

} // namespace ZJVIDEO
