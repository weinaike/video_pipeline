#include "BaseNode.h"

namespace ZJVIDEO{

BaseNode::BaseNode(const NodeParam & param) : AbstractNode(param)
{
    // 注册日志记录器
    el::Loggers::getLogger(BASENODE_LOG);


    m_node_type = param.m_node_type;
    m_name = param.m_node_name;
    m_channels = param.m_channels;
    m_cfg_file = param.m_cfg_file;
    parse_configure(m_cfg_file);
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

int BaseNode::connect_add_input(const std::string & tag, std::shared_ptr<ThreadSaveQueue> queue) 
{

    std::unique_lock<std::mutex> lk(m_base_mutex);
    queue->setCond(m_base_cond);
    m_input_buffers.insert(make_pair(tag, queue));

    return ZJV_STATUS_OK;
}

int BaseNode::connect_add_output(const std::string & tag, std::shared_ptr<ThreadSaveQueue> queue)
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
    m_run = true;
    m_worker = std::thread(&BaseNode::worker, this);
    CLOG(INFO, BASENODE_LOG)<<m_name<<" 线程启动";
    return ZJV_STATUS_OK;
}


int BaseNode::stop()
{
    if (!m_run) 
    {
        CLOG(INFO, BASENODE_LOG)<< m_name << " tread exited already";
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
    CLOG(INFO, BASENODE_LOG)<<m_name<<" thread exit success";
    return ZJV_STATUS_OK;
}

int BaseNode::worker()
{
    std::vector<std::shared_ptr<BaseData>> datas;
    while (m_run) 
    {
        datas.clear();
        // 提取数据
        for (auto &item : m_input_buffers) 
        {
            std::shared_ptr<BaseData> data;
            int count = 0;
            while (item.second->Pop(data)) 
            {
                datas.push_back(data);
                count++;
                if (count >= m_get_data_max_num) 
                {
                    break;
                }
            }
        }

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
            for (auto &data : datas) 
            {
                for (auto &item : m_output_buffers) 
                {
                    item.second->Push(data);
                }
            }
        } 

    }
    return ZJV_STATUS_OK;
}

int BaseNode::process(std::vector<std::shared_ptr<BaseData>> & data)
{
    return ZJV_STATUS_OK;
}

bool BaseNode::get_run_status()
{
    return m_run;
}
std::string BaseNode::get_name()
{
    return m_name;
}

REGISTER_NODE_CLASS(Base)

} // namespace ZJVIDEO