

#include "ImageCacheNode.h"
#include "nlohmann/json.hpp"
#define CACHE_LOG "ImageCache"
namespace ZJVIDEO {

ImageCacheNode::ImageCacheNode(const NodeParam & param) : BaseNode(param)
{
    m_logger = el::Loggers::getLogger(CACHE_LOG);
    el::Configurations conf;
    conf.setToDefault();
    // Get the format for Info level
    std::string infoFormat = conf.get(el::Level::Info, el::ConfigurationType::Format)->value();
    // Set the format for Debug level to be the same as Info level
    conf.set(el::Level::Debug, el::ConfigurationType::Format, infoFormat);
    el::Loggers::reconfigureLogger(m_logger, conf);

    m_batch_process = false;
    m_max_batch_size = 1;
    m_count = 0;
    m_append_count = 0;
    parse_configure(param.m_cfg_file);
}

ImageCacheNode::~ImageCacheNode()
{
    CLOG(INFO, CACHE_LOG) << "ImageCacheNode::~ImageCacheNode";
}

int ImageCacheNode::parse_configure(std::string cfg_file)
{

    CLOG(INFO, CACHE_LOG) << "ImageCacheNode::parse_configure";
    std::ifstream i(cfg_file);
    if(i.is_open() == false)
    {
        CLOG(ERROR, CACHE_LOG) << "open cfg_file failed";
        m_os_type = 0;
        m_width = 640;
        m_height = 480;
        m_frame_num = 16;
        m_step = 0;
        m_fps = 1.0;

        return ZJV_STATUS_ERROR;
    }
    nlohmann::json j;
    i >> j;
    std::string type = j["type"];
    if(type == "continue")
    {
        m_os_type = ZJV_OUTPUT_STREAM_TYPE_CONTINUE;
    }
    else if(type == "trigger")
    {
        m_os_type = ZJV_OUTPUT_STREAM_TYPE_TRIGGER;
    }
    else
    {
        m_os_type = ZJV_OUTPUT_STREAM_TYPE_CONTINUE;
    }
    m_width = j["width"];
    m_height = j["height"];
    m_frame_num = j["num"];
    m_step = j["step"];
    m_fps = j["fps"];

    // 打印配置参数
    CLOG(INFO, CACHE_LOG) << "----------------ImageCacheNode config-----------------";
    CLOG(INFO, CACHE_LOG) << "type:    [" << m_os_type << "]";
    CLOG(INFO, CACHE_LOG) << "width:   [" << m_width << "]";
    CLOG(INFO, CACHE_LOG) << "height:  [" << m_height << "]";
    CLOG(INFO, CACHE_LOG) << "num:     [" << m_frame_num << "]";
    CLOG(INFO, CACHE_LOG) << "step:    [" << m_step << "]";
    CLOG(INFO, CACHE_LOG) << "fps:     [" << m_fps << "]";
    CLOG(INFO, CACHE_LOG) << "----------------ImageCacheNode config-----------------";

    return 0;
}

int ImageCacheNode::init()
{
    CLOG(INFO, CACHE_LOG) << "ImageCacheNode::init";
    return 0;
}

int ImageCacheNode::process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas)
{
    int output_interval_num = 1;
    std::shared_ptr<const FrameData> in_frame_data = nullptr;
    for(const auto & in :in_metas)
    {
        if (in->data_name == "Frame")
        {            
            in_frame_data = std::dynamic_pointer_cast<const FrameData>(in);
            output_interval_num = in_frame_data->fps/m_fps > output_interval_num ? in_frame_data->fps/m_fps : output_interval_num;
        }
        else
        {
            // trigger condition
        }
    }
    // CLOG(DEBUG, CACHE_LOG) << "output_interval_num: " << output_interval_num << "  frame_id : " << in_frame_data->frame_id
    //                         << " m_step: " << m_step << " m_count: " << m_count << " m_append_count: " << m_append_count 
    //                         << " m_frame_datas.size(): " << m_frame_datas.size();
    
    if(m_append_count % (m_step + 1) == 0)
    {    
        m_count++;
        // add to list
        std::shared_ptr<FrameData> frame_data = std::make_shared<FrameData>(*in_frame_data);
        m_frame_datas.push_back(frame_data);    
        m_append_count = 0;

        if(m_frame_datas.size() > 32)
        {
            m_frame_datas.pop_front();
        }
    }
    m_append_count++;

    // output
    if(m_os_type == ZJV_OUTPUT_STREAM_TYPE_CONTINUE)
    {
        std::shared_ptr<ImageCahceData> out = std::make_shared<ImageCahceData>();
        if(m_frame_datas.size() >= m_frame_num && m_count >= output_interval_num)
        {
            auto it = m_frame_datas.begin();
            for(int i  = 0 ; i < m_frame_num; i++)
            {
                //std::cout << " " << (*it)->frame_id;
                out->images.push_back(*it);
                it++;
            }
            //std::cout << std::endl;
            m_count = 0;
            // CLOG(DEBUG, CACHE_LOG) << "output cache: " << out->images.size();
        }
        out_metas.push_back(out);
    }
    else
    {
        std::shared_ptr<ImageCahceData> out = std::make_shared<ImageCahceData>();
        // trigger condition

        out_metas.push_back(out);
    }


    // CLOG(INFO, CACHE_LOG) << "ImageCacheNode::process_single";
    return 0;
}

REGISTER_NODE_CLASS(ImageCache)

} // namespace ZJVIDEO
