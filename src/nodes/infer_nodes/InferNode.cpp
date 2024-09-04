

#include "InferNode.h"

namespace ZJVIDEO {



InferNode::InferNode(const NodeParam & param) : BaseNode(param)
{
    m_logger = el::Loggers::getLogger(INFER_LOG);
    el::Configurations conf;
    conf.setToDefault();
    // Get the format for Info level
    std::string infoFormat = conf.get(el::Level::Info, el::ConfigurationType::Format)->value();
    // Set the format for Debug level to be the same as Info level
    conf.set(el::Level::Debug, el::ConfigurationType::Format, infoFormat);
    el::Loggers::reconfigureLogger(m_logger, conf);
    parse_configure(param.m_cfg_file);

    init();
    if(m_engine_param.m_max_batch_size > 8)
    {
        m_max_batch_size = m_engine_param.m_max_batch_size; // 根据模型配置设置
    }
    
    m_blob_input_flag = false;
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
            m_engine_param.m_device_id = j["model"]["device"];
            m_engine_param.m_int8 = j["model"]["enable_int8"];
            m_engine_param.m_fp16 = j["model"]["enable_fp16"];
            
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
    
    m_device_id = m_engine_param.m_device_id;
    // 2. 解析出 preprocess
    int lib_type = ZJV_PREPROCESS_LIB_CIMG;
    if(m_engine_param.m_device_id >= 0 )
    {
        lib_type = ZJV_PREPROCESS_LIB_CUDA;
    }

    if(j.contains("preprocess"))
    {
        if(j["preprocess"].contains("ImageType"))
        {
            nlohmann::json image_types = j["preprocess"]["ImageType"];
            for(auto & image_type: image_types)
            {
                std::shared_ptr<PreProcessor> img_preproc = std::make_shared<PreProcessor>(lib_type, m_device_id);
                img_preproc->parse_json(image_type);
                PreProcessParameter param = img_preproc->get_param();
                m_img_preprocs.push_back(img_preproc);
                m_img_preproc_params.push_back(param);
            }
        }
        
        if(j["preprocess"].contains("VideoType"))
        {
            nlohmann::json video_types = j["preprocess"]["VideoType"];
            for(auto & video_type: video_types)
            {
                std::shared_ptr<PreProcessor> video_preproc = std::make_shared<PreProcessor>(lib_type, m_device_id);
                video_preproc->parse_json(video_type);
                PreProcessParameter param = video_preproc->get_param();
                m_img_preprocs.push_back(video_preproc);
                m_img_preproc_params.push_back(param);
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

    float pre_time = 0;
    float infer_time = 0;
    float post_time = 0;


    for(int i = 0; i < frame_rois.size(); i += m_max_batch_size)
    {
        std::vector<std::shared_ptr<FrameROI>> batch_frame_rois;

        for(int j = 0; j < m_max_batch_size; j++)
        {
            if((i+j) >= frame_rois.size()) break;
            batch_frame_rois.push_back(frame_rois[i+j]);
        }

        auto t1 = std::chrono::system_clock::now();
        std::vector<FBlob> inputs;
        preprocess(batch_frame_rois, inputs);
        auto t2 = std::chrono::system_clock::now();
        std::vector<FBlob> outputs;
        infer(inputs, outputs);
        auto t3 = std::chrono::system_clock::now();
        postprocess(outputs, batch_frame_rois);
        auto t4 = std::chrono::system_clock::now();

        std::chrono::duration<double> dt1 = t2 - t1; // Calculate elapsed time
        std::chrono::duration<double> dt2 = t3 - t2; // Calculate elapsed time
        std::chrono::duration<double> dt3 = t4 - t3; // Calculate elapsed time
        pre_time +=  dt1.count() ; 
        infer_time +=  dt2.count() ;
        post_time +=  dt3.count() ;
    }

    auto t5 = std::chrono::system_clock::now();
    summary(frame_rois, out_metas_batch);
    auto t6 = std::chrono::system_clock::now();
    std::chrono::duration<double> dt4 = t6 - t5; // Calculate elapsed time


    CLOG(TRACE, INFER_LOG) <<"cost time: size["<< frame_rois.size()<< "] preprocess: " << pre_time/frame_rois.size() * 1000 << 
                            "ms infer: " << infer_time/frame_rois.size() * 1000 << 
                            "ms postprocess: " << post_time/frame_rois.size() * 1000<< 
                            "ms summary: " << dt4.count() * 1000 << "ms";

    return ZJV_STATUS_OK;
}

int InferNode::prepare( const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                            std::vector<std::shared_ptr<FrameROI>>  &frame_rois)
{
    if(in_metas_batch.size() == 0) return ZJV_STATUS_ERROR;

    // 遍历 m_nodeparam.m_input_node_datas
    bool is_frame = false;
    bool is_image_cache = false;
    bool is_blob = false;
    for (size_t i = 0; i < m_nodeparam.m_input_node_datas.size(); i++)
    {
        //    std::vector<std::pair<std::string, std::string>>    m_input_node_datas;   // 前置节点数据
        std::string key = m_nodeparam.m_input_node_datas[i].first;
        std::string value = m_nodeparam.m_input_node_datas[i].second;
        if(value == "Frame")
        {
            is_frame = true;
        }
        if(value == "ImageCache")
        {
            is_image_cache = true;
        }
        if(value == "FeatureCache")
        {
            is_blob = true;
            m_blob_input_flag = true;
        }
    }
  
    if(is_frame)
    {
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
                for(int j = 0; j < in_metas_batch[i].size(); j++)
                {                
                    if(in_metas_batch[i][j]->data_name == "Frame")
                    {
                        frame_data = std::dynamic_pointer_cast<const FrameData>(in_metas_batch[i][j]);                  
                    }
                }

                for(int j = 0; j < in_metas_batch[i].size(); j++)
                {                
                    if(in_metas_batch[i][j]->data_name == "DetectResult")
                    {
                        std::shared_ptr<const DetectResultData> roi_data = std::dynamic_pointer_cast<const DetectResultData>(in_metas_batch[i][j]);
                        for(int k = 0; k < roi_data->detect_boxes.size(); k++)
                        {
                            std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
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
                }
                
            }
        }
    }

    if(is_image_cache)
    {
        for(int i = 0; i < in_metas_batch.size(); i++)
        {
            std::shared_ptr<const ImageCahceData> cache = nullptr;
            bool has_cache = false;
            bool has_roi = false;
            for(int j = 0; j < in_metas_batch[i].size(); j++)
            {            
                if(in_metas_batch[i][j]->data_name == "ImageCache")
                {
                    cache = std::dynamic_pointer_cast<const ImageCahceData>(in_metas_batch[i][j]);
                    if(cache->images.size() > 0)
                    {
                        has_cache = true;
                    }                
                }
            }

            if (has_cache)
            {
                for(int j = 0; j < in_metas_batch[i].size(); j++)
                {            
                    if(in_metas_batch[i][j]->data_name == "DetectResult")
                    {                    
                        std::shared_ptr<const DetectResultData> roi_data = std::dynamic_pointer_cast<const DetectResultData>(in_metas_batch[i][j]);
                        for(int k = 0; k < roi_data->detect_boxes.size(); k++)
                        {                        
                            Rect roi;
                            roi.x = roi_data->detect_boxes[k].x1;
                            roi.y = roi_data->detect_boxes[k].y1;
                            roi.width = (roi_data->detect_boxes[k].x2 - roi_data->detect_boxes[k].x1)/2*2;
                            roi.height = (roi_data->detect_boxes[k].y2 - roi_data->detect_boxes[k].y1)/2*2;

                            std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
                            
                            frame_roi->roi = roi;
                            frame_roi->input_vector_id = i;
                            frame_roi->original = &(roi_data->detect_boxes[k]);
                            for(int m = 0; m < cache->images.size(); m++)
                            {
                                frame_roi->frames.push_back(cache->images[m]);
                            }
                            frame_rois.push_back(frame_roi);   
                        }
                        has_roi = true;
                    }
                }
                if(!has_roi)
                {
                    if(cache == nullptr)
                    {
                        CLOG(ERROR, INFER_LOG) << "cache is nullptr";
                    }
                    std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
                    frame_roi->input_vector_id = i;
                    frame_roi->roi.x = 0;
                    frame_roi->roi.y = 0;
                    frame_roi->roi.width = cache->images[0]->width/2*2 ;
                    frame_roi->roi.height = cache->images[0]->height/2*2 ;
                    frame_roi->original = nullptr;
                    
                    for(int j = 0; j < cache->images.size(); j++)
                    {
                        frame_roi->frames.push_back(cache->images[j]);
                    }

                    frame_rois.push_back(frame_roi);
                }                    
            }
        }
    }

    if(is_blob)
    {
        for(int i = 0; i < in_metas_batch.size(); i++)
        {
            std::shared_ptr<const FeatureCacheData> blob = nullptr;
            bool has_blob = false;
            for(int j = 0; j < in_metas_batch[i].size(); j++)
            {            
                if(in_metas_batch[i][j]->data_name == "FeatureCache")
                {
                    blob = std::dynamic_pointer_cast<const FeatureCacheData>(in_metas_batch[i][j]);
                    if(blob->feature)
                    {
                        has_blob = true;
                    }                
                }
            }

            if (has_blob)
            {               
                if(blob == nullptr)
                {
                    CLOG(ERROR, INFER_LOG) << "blob is nullptr";
                }
                std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
                frame_roi->input_vector_id = i;

                frame_roi->roi.x = 0;
                frame_roi->roi.y = 0;
                frame_roi->roi.width = blob->feature->shape()[3]/2*2 ;
                frame_roi->roi.height = blob->feature->shape()[2]/2*2 ;
                frame_roi->original = nullptr;

                frame_roi->fblob = blob->feature;
                frame_rois.push_back(frame_roi);
                                  
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
        if(m_blob_input_flag)
        {
            float * data = input_blob.mutable_cpu_data();
            for(int j = 0; j < frame_rois.size(); j++)
            {
                int size = frame_rois[j]->fblob->count();
                memcpy(data + j * size, frame_rois[j]->fblob->cpu_data(), size * sizeof(float));
            }
        }
        else
        {
            m_img_preprocs[i]->run(frame_rois, input_blob, m_img_preproc_params[i]);
        }        
        inputs.push_back(input_blob);
    }

    return ZJV_STATUS_OK;
}
int InferNode::infer(std::vector<FBlob> & inputs, std::vector<FBlob> & outputs)
{   

    m_engine->forward(inputs, outputs); 
    
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
