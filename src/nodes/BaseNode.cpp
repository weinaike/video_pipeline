#include "BaseNode.h"

namespace ZJVIDEO{

BaseNode::BaseNode(const NodeParam & param) : AbstractNode(param),
    m_nodeparam(param)
{
    // 注册日志记录器
    el::Loggers::getLogger(BASENODE_LOG);
    m_wait_data = param.m_wait_data;
    m_batch_process = param.m_channels;
    for (const auto & item : m_nodeparam.m_input_node_datas) 
    {
        if(item.second == "Frame")
        {
            continue;
        }
        std::string extra_nmae= item.first + "_" + item.second;
        m_input_data_names.push_back(extra_nmae);
    }

    // 输出节点信息
    CLOG(INFO, BASENODE_LOG) << "node name: " << m_nodeparam.m_node_name;
    CLOG(INFO, BASENODE_LOG) << "node type: " << m_nodeparam.m_node_type;
    CLOG(INFO, BASENODE_LOG) << "node channels: " << m_nodeparam.m_channels;
    CLOG(INFO, BASENODE_LOG) << "node cfg file: " << m_nodeparam.m_cfg_file;
    CLOG(INFO, BASENODE_LOG) << "node channel id: " << m_nodeparam.m_channel_id;
    CLOG(INFO, BASENODE_LOG) << "node output datas: " << m_nodeparam.m_output_datas.size();
    for (const auto & item : m_nodeparam.m_output_datas) 
    {
        CLOG(INFO, BASENODE_LOG) << "output data: " << item;
    }
    CLOG(INFO, BASENODE_LOG) << "node input node datas: " << m_nodeparam.m_input_node_datas.size();
    for (const auto & item : m_nodeparam.m_input_node_datas) 
    {
        CLOG(INFO, BASENODE_LOG) << "input node data: " << item.first << " " << item.second;
    }
    CLOG(INFO, BASENODE_LOG) << "node wait data: " << m_nodeparam.m_wait_data;
    CLOG(INFO, BASENODE_LOG) << "node batch process: " << m_nodeparam.m_channels;



    parse_configure(m_nodeparam.m_cfg_file);
    init();
}

BaseNode::~BaseNode()
{
    stop();
}

int BaseNode::parse_configure(std::string cfg_file)
{
    return ZJV_STATUS_OK;
} 

int BaseNode::init()
{
    
    return ZJV_STATUS_OK;
}

int BaseNode::connect_add_input(const std::string & tag, std::shared_ptr<FlowQueue> queue) 
{

    std::unique_lock<std::mutex> lk(m_base_mutex);
    queue->setCond(m_base_cond);
    m_input_buffers.insert(make_pair(tag, queue));

    return ZJV_STATUS_OK;
}

int BaseNode::connect_add_output(const std::string & tag, std::shared_ptr<FlowQueue> queue)
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_output_buffers.insert(make_pair(tag, queue));

    return ZJV_STATUS_OK;
}


int BaseNode::disconnect_del_input(const std::string & tag)
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_input_buffers.find(tag) != m_input_buffers.end()) 
    {
        m_input_buffers.erase(tag);
    }
    return ZJV_STATUS_OK;
}

int BaseNode::disconnect_del_output(const std::string & tag)
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_output_buffers.find(tag) != m_output_buffers.end()) 
    {
        m_output_buffers.erase(tag);
    }
    return ZJV_STATUS_OK;
}

int BaseNode::start()
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_run) 
    {
        CLOG(INFO, BASENODE_LOG)<<"该线程重复启动";
        return ZJV_STATUS_OK;
    }
    if(m_input_buffers.size() == 0 || m_output_buffers.size() == 0)
    {
        CLOG(ERROR, BASENODE_LOG)<<"无输入或者输出队列";
        return ZJV_STATUS_ERROR;
    }

    // 节点输入需求，与输入队列是否已准备就绪
    if(m_nodeparam.m_input_node_datas.size() == 0)
    {
        m_node_position_type = ZJV_NODE_POSITION_SRC;
    }
    for(const auto & output : m_output_buffers)
    {
        if(output.first == m_nodeparam.m_node_name)
        {
             m_node_position_type = ZJV_NODE_POSITION_DST;
        }
    }

    for(const auto & item : m_nodeparam.m_input_node_datas)
    {
        if (m_input_buffers.find(item.first) == m_input_buffers.end()) 
        {
            CLOG(ERROR, BASENODE_LOG) << "input queue ["<< item.first <<"] is not ready in " << m_nodeparam.m_node_name ;
            return ZJV_STATUS_ERROR;
        }
    }
    if(m_node_position_type == ZJV_NODE_POSITION_UNKNOWN)
    {
        m_node_position_type = ZJV_NODE_POSITION_MID;
    }

    CLOG(INFO, BASENODE_LOG) <<  m_nodeparam.m_node_name << " type :" << m_node_position_type ;

    CLOG(INFO, BASENODE_LOG) <<  m_nodeparam.m_node_name << " input queue size :" << m_input_buffers.size() ;
    CLOG(INFO, BASENODE_LOG) <<  m_nodeparam.m_node_name << " output queue size :" << m_output_buffers.size() ;


    m_run = true;
    m_worker = std::thread(&BaseNode::worker, this);
    
    CLOG(INFO, BASENODE_LOG)<< m_nodeparam.m_node_name <<" thread start";
    return ZJV_STATUS_OK;
}


int BaseNode::stop()
{
    if (!m_run) 
    {
        CLOG(INFO, BASENODE_LOG)<< m_nodeparam.m_node_name << " tread exited already";
        return ZJV_STATUS_OK;
    }
    m_run = false;

    // 清空队列
    for (auto & it : m_input_buffers) 
    {
        it.second->clear();
    }
    for (auto & it : m_output_buffers) 
    {
        it.second->clear();
    }

    m_base_cond->notify_all();
    if (m_worker.joinable()) 
    {
        m_worker.join();
    }
    CLOG(INFO, BASENODE_LOG)<< m_nodeparam.m_node_name <<" thread exit success";
    return ZJV_STATUS_OK;
}


// 收集1各通道的一组数据（该组数据包含所有输入需求）
int BaseNode::get_input_data(std::vector<std::shared_ptr< FlowData>> &datas)
{
    int cnt = 0;

    // 单输入队列
    if(m_input_buffers.size() <= 1)
    {
        // std::cout <<m_nodeparam.m_node_name << "单输入队列"<<std::endl;
        for (const auto & buffer :m_input_buffers)
        {
            std::shared_ptr<FlowData> data;
            if(buffer.second->Pop(data))
            {
                datas.push_back(data);
                cnt++;
                if(cnt >= m_get_data_max_num)
                {
                    break;
                }
            }
            else
            {
                // CLOG(INFO,BASENODE_LOG) <<m_nodeparam.m_node_name << "queue empty";
                break;
            }
        }
        
    }
    else // 多输入情况，先遍历队列，均摊负载
    {
        std::cout<<m_nodeparam.m_node_name << " multi channels"<<std::endl;
        while(cnt < m_get_data_max_num)
        {
            for (const auto & buffer :m_input_buffers)
            {
                std::shared_ptr<FlowData> data;
                if(buffer.second->Pop(data))
                {
                    // 判断 data 是否满足接收条件, 不满足，直接丢弃
                    if(data->has_extras(m_input_data_names)) 
                    {
                        datas.push_back(data);
                        cnt++;
                        // 遍历所有同通道队列，去掉同一帧数据，避免重复计算
                        int id = data->get_channel_id() ;
                        
                        for (const auto & buffer :m_input_buffers)
                        {
                            if( id == parse_id(buffer.first))
                            {
                                std::shared_ptr<FlowData> temp;
                                buffer.second->front(temp);
                                if(temp->frame->frame_id == data->frame->frame_id )
                                {
                                    buffer.second->Pop(temp);
                                }
                            }
                        }

                    }
                    else
                    {
                        // CLOG(DEBUG,BASENODE_LOG) <<m_nodeparam.m_node_name << "has_extras is false";
                    }
                }
            }
        }
        
    }



    return ZJV_STATUS_OK;
}
int BaseNode::send_output_data(const std::vector<std::shared_ptr<FlowData>> &datas)
{
    if(m_output_buffers.size() == 1)
    {
        // 不支持多通道的，直接发送
        for (const auto & item : m_output_buffers) 
        {
            for (const auto & data : datas) 
            {
                item.second->Push(data);
            }
        }
    }
    else
    {
        // 支撑多通道的， 按通道分发
        if(m_nodeparam.m_channels)
        {
            for (const auto & data : datas) 
            {
                for (const auto & item : m_output_buffers) 
                {
                    int id = parse_id(item.first);
                    // 发送到对应管道
                    if(id == data->get_channel_id())
                    {
                        item.second->Push(data);
                    }                    
                }
            }
        }
        else 
        {
            // 不支持多通道的，直接发送
            for (const auto & item : m_output_buffers) 
            {
                for (const auto & data : datas) 
                {
                    item.second->Push(data);
                }
            }
        }
    }

    return ZJV_STATUS_OK;

}


int BaseNode::worker()
{
    std::vector<std::shared_ptr<FlowData>> datas;
    while (m_run) 
    {
        datas.clear();
        get_input_data(datas);

        if (!datas.empty()) 
        {
            std::unique_lock<std::mutex> lk(m_base_mutex);
            m_base_cond->wait(lk);
            continue;
        }
        else
        {
            // 主处理
            process(datas);
            // 传递数据
            send_output_data(datas);
        } 

    }
    return ZJV_STATUS_OK;
}

int BaseNode::process(const std::vector<std::shared_ptr<FlowData>> & datas)
{
    for(const auto & data :datas)
    {
        std::vector<std::pair<std::string, std::shared_ptr<const ExtraData> > > metas;

        for (const auto & output_data : m_nodeparam.m_output_datas)
        {
            std::string name = m_nodeparam.m_node_name + "_"+output_data;
            std::shared_ptr<const ExtraData> meta = std::make_shared<ExtraData>();
            metas.push_back({name, meta});
        }
        data->push_back(metas);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return ZJV_STATUS_OK;
}

bool BaseNode::get_run_status()
{
    return m_run;
}
std::string BaseNode::get_name()
{
    return m_nodeparam.m_node_name;
}

REGISTER_NODE_CLASS(Base)

} // namespace ZJVIDEO