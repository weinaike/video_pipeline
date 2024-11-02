

#include "WeldEventNode.h"
#include "nlohmann/json.hpp"
#include "common/ExtraData.h"

#define WELD_LOG "Weld"
namespace ZJVIDEO {

WeldEventNode::WeldEventNode(const NodeParam & param) : BaseNode(param)
{
    m_logger = el::Loggers::getLogger(WELD_LOG);
    el::Configurations conf;
    conf.setToDefault();
    // Get the format for Info level
    std::string infoFormat = conf.get(el::Level::Info, el::ConfigurationType::Format)->value();
    // Set the format for Debug level to be the same as Info level
    conf.set(el::Level::Debug, el::ConfigurationType::Format, infoFormat);
    el::Loggers::reconfigureLogger(m_logger, conf);

    m_batch_process = false;
    m_max_batch_size = 1;

    parse_configure(param.m_cfg_file);
}

WeldEventNode::~WeldEventNode()
{
    CLOG(INFO, WELD_LOG) << "WeldEventNode::~WeldEventNode";
}

int WeldEventNode::parse_configure(std::string cfg_file)
{
    CLOG(INFO, WELD_LOG) << "WeldEventNode::parse_configure";   
    return 0;
}

int WeldEventNode::init()
{
    CLOG(INFO, WELD_LOG) << "WeldEventNode::init";
    return 0;
}


int WeldEventNode::process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas)
{
    std::shared_ptr<WeldResultData> event = std::make_shared<WeldResultData>();
    event->is_enable = false;

    for(const auto & in :in_metas)
    {
        if(in->data_name == "Frame")
        {
            std::shared_ptr<const FrameData> frame_data = std::dynamic_pointer_cast<const FrameData>(in);
            event->camera_id = frame_data->camera_id;
            event->frame_id = frame_data->frame_id;
        }
        else if(in->data_name == "ClassifyResult")
        {            
            std::shared_ptr<const ClassifyResultData> classify_data = std::dynamic_pointer_cast<const ClassifyResultData>(in);
            for(int i = 0; i < classify_data->obj_attr_info.size(); i++)
            {
                event->is_enable = true;
                ObjectAttribute obj_attr = classify_data->obj_attr_info[i];
                // std::cout << obj_attr.attribute << " " << obj_attr.attr_sub_category << " " << obj_attr.score << std::endl;
                if(obj_attr.attribute == ZJV_ATTRIBUTE_CATEGORY_WELDSEAM_STATUS)
                {
                    event->weld_status = obj_attr.attr_sub_category;
                    event->status_score = obj_attr.score;                    
                }
                else if(obj_attr.attribute == ZJV_ATTRIBUTE_VALUE_WELDSEAM_DEPTH)
                {
                    event->weld_depth = obj_attr.attr_value;
                }
                else if(obj_attr.attribute == ZJV_ATTRIBUTE_VALUE_WELDSEAM_FRONT_QUALITY)
                {
                    event->front_quality = obj_attr.attr_value;
                }
                else if(obj_attr.attribute == ZJV_ATTRIBUTE_VALUE_WELDSEAM_BACK_QUALITY)
                {
                    event->back_quality = obj_attr.attr_value;
                }
            }
            
        }
    }
    if(event->is_enable)
    {
        m_status.push_back(event->weld_status);
        m_depth.push_back(event->weld_depth);
        m_front.push_back(event->front_quality);
        m_back.push_back(event->back_quality);
        m_score.push_back(event->status_score);

        if(m_status.size() > m_max_size)
        {
            m_status.pop_front();
            m_depth.pop_front();
            m_front.pop_front();
            m_back.pop_front();
            m_score.pop_front();
        }
    }

    // 平滑status
    if(event->is_enable)
    {        
        int num = 4;
        if(m_status.size() >= num)
        {
            int start = m_status.size() - num;
            std::map<int, float> map;
            for(int s = start; s < m_status.size(); s++)
            {
                bool found = false;
                for (auto& pair : map)
                {
                    if (pair.first == m_status[s])
                    {
                        map[pair.first] += m_score[s];
                        found = true;
                        break;
                    }
                }
                if(!found)
                {
                    map[m_status[s]] = m_score[s];
                }                
            }
            int label = -1;
            float max_score = 0;
            for (const auto& pair : map)
            {
                if(pair.second > max_score)
                {
                    label = pair.first;
                    max_score = pair.second;
                }
            }
            event->weld_status = label;            
        }
    }
    // 平滑depth、front、back
    if(event->is_enable)
    {
        int num = 5;
        if(m_depth.size() >= num)
        {
            int start = m_depth.size() - num;
            float depth = 0;
            float front = 0;
            float back = 0;
            for(int s = start; s < m_depth.size(); s++)
            {
                depth += m_depth[s];
                front += m_front[s];
                back += m_back[s];
            }
            event->weld_depth = depth / num;
            event->front_quality = front / num;
            event->back_quality = back / num;
        }
    }
    out_metas.push_back(event);

    return 0;
}


int WeldEventNode::control(std::shared_ptr<ControlData>& data)
{
    CLOG(INFO, WELD_LOG) << "CacheNode::control";
    // 调用基类的control函数
    BaseNode::control(data);
    if(data->get_control_type() == ZJV_CONTROLTYPE_CLEAR_CACHE)
    {
        m_status.clear();
        m_score.clear();
        m_depth.clear();
        m_front.clear();
        m_back.clear();
    }

    return 0;
}


REGISTER_NODE_CLASS(WeldEvent)

} // namespace ZJVIDEO
