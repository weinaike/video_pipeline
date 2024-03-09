//
// Created by lijin on 2023/12/6.
//

#include "Pipeline.h"
#include <../nlohmann/json.hpp>
#include <fstream>

namespace ZJVIDEO {


Pipeline::Pipeline(std::string cfg_file) 
{
    el::Loggers::getLogger(PIPE_LOG);
    parse_cfg_file(cfg_file);
    m_initialized = false;
    CLOG(INFO, PIPE_LOG) << "Pipeline created" ;
}


int Pipeline::start() 
{
    if (!m_initialized) 
    {
        CLOG(INFO, PIPE_LOG) << "Pipeline not initialized" ;

        return ZJV_STATUS_ERROR;
    }

    for (auto & node : m_node_map) 
    {
        int ret = node.second->start();
        if (ret != ZJV_STATUS_OK) 
        {
            CLOG(INFO, PIPE_LOG) << "Node " << node.first << " start failed" ;
        }
    }

    return ZJV_STATUS_OK;
}

int Pipeline::stop() 
{
    if(m_initialized)
    {
        for (auto & node : m_node_map) 
        {
            if(node.second->get_run_status())
            {
                node.second->stop();
            }        
        }
    }
    else
    {
        CLOG(INFO, PIPE_LOG) << "Pipeline not initialized" ;
    }

    return ZJV_STATUS_OK;
}

Pipeline::~Pipeline() {
    stop();
}

struct DataParam {
    std::string pre_node;
    std::string meta_data_type;
};
static void from_json(const nlohmann::json& j, DataParam& p) 
{   
    if(j.contains("pre_node"))
    {
        j.at("pre_node").get_to(p.pre_node);
    }
    if(j.contains("meta_data_type"))
    {
        j.at("meta_data_type").get_to(p.meta_data_type);
    }
}

static void from_json(const nlohmann::json& j, NodeParam& p) 
{
    j.at("node_type").get_to(p.m_node_type);
    j.at("node_name").get_to(p.m_node_name);
    j.at("channels").get_to(p.m_channels);
    j.at("cfg_file").get_to(p.m_cfg_file);
    if (j.contains("input_datas")) 
    {
        for (const auto& element : j.at("input_datas")) 
        {
            DataParam data;
            from_json(element, data);
            p.m_input_node_datas.push_back(std::make_pair(data.pre_node,data.meta_data_type));
        }
    }
    if (j.contains("output_datas")) 
    {
        for (const auto& element : j.at("output_datas")) 
        {
            DataParam data;
            from_json(element, data);
            p.m_output_datas.push_back(data.meta_data_type);
        }
    }
}

int Pipeline::expand_pipe() 
{
    // 在所有节点中，找出支持多通道的节点
    std::vector<NodeParam> m_multi_channel_nodes;
    for (auto & node : m_nodeparams) 
    {
        if(node.m_channels)
        {
            NodeParam node_copy = node;
            m_multi_channel_nodes.push_back(node_copy);
        }
    }

    // 多通道，节点扩展为多份
    
    for(int i = 0; i < m_channel_num; i++)
    {
        for (auto & node : m_nodeparams) 
        {
            NodeParam node_copy = node;
            node_copy.m_node_name = join_string(node.m_node_name , i);
            for(auto & input: node_copy.m_input_node_datas)
            {
                input.first =  join_string(input.first , i);
            }
            m_channels[i].push_back(node_copy);
        }
    }
    // 多通道，节点链接也相应扩展多份
    std::map<int, std::vector<std::pair<std::string, std::string>>> m_channels_connect_list;
    for(int i = 0; i < m_channel_num; i++)
    {
        for (auto & connection : m_connect_list) 
        {
            std::pair<std::string, std::string> connection_copy = connection;
            connection_copy.first = join_string(connection.first , i);
            connection_copy.second = join_string(connection.second , i);
            m_channels_connect_list[i].push_back(connection_copy);
        }
    }

    // 对于支持多通道的节点， 将扩展后的节点和链接关系，合并到多通道节点中
    for (auto & node : m_multi_channel_nodes) 
    {
        for(int i = 0; i < m_channel_num; i++)
        {
            // 删除支持多通道的节点
            for (auto & item : m_channels[i]) 
            {
                std::string temp_name = join_string(node.m_node_name,i);
                for(auto & input :item.m_input_node_datas)
                {
                    if (input.first == temp_name)
                    {
                        input.first = node.m_node_name;
                    }
                }                
                m_channels[i].erase(std::remove_if(m_channels[i].begin(), m_channels[i].end(), [&](const NodeParam& item) {
                    return item.m_node_name == temp_name;
                }), m_channels[i].end());
            }

            // 对于支持多通道的节点名，扩展后节点名称替换为原名
            for (auto & item : m_channels_connect_list[i]) 
            {
                std::string temp_name = join_string(node.m_node_name,i);
                if(item.first == temp_name)
                {
                    item.first = node.m_node_name;
                }
                if(item.second == temp_name)
                {
                    item.second = node.m_node_name;
                }
            }
        }
    }

    // 清空，m_nodeparams
    m_nodeparams.clear();
    // 通道节点合并入m_nodeparams
    for(int i = 0; i < m_channel_num; i++)
    {
        for (auto & node : m_channels[i]) 
        {
            node.m_channel_id = i;
            m_nodeparams.push_back(node);
        }
    }

    std::vector<std::string> multi_names;
    for (auto & node : m_multi_channel_nodes) 
    {
        multi_names.push_back(node.m_node_name);
    }

    for (auto & node : m_multi_channel_nodes) 
    {
        for(auto & input : node.m_input_node_datas)
        {
            // 判断 input.first 是否在multi_names 
            if (std::find(multi_names.begin(), multi_names.end(), input.first) == multi_names.end())
            {
                std::string name = input.first;
                std::string type = input.second;
                input.first = join_string(name, 0);
                for(int i = 1; i < m_channel_num; i++)
                {
                    node.m_input_node_datas.push_back({join_string(name, i), type});
                    //std::cout<<node.m_node_name<<" " <<join_string(name, i)<< " type "<<type<<std::endl;
                }
            }
        }
    }

    // 添加多通道节点
    for (auto & node : m_multi_channel_nodes) 
    {
        m_nodeparams.push_back(node);
    }

    // 清空，m_connect_list
    m_connect_list.clear();
    // 通道链接合并如m_connect_list, <first, next>相同的不重复添加
    for(int i = 0; i < m_channel_num; i++)
    {
        for (auto & connection : m_channels_connect_list[i]) 
        {
            if(std::find(m_connect_list.begin(), m_connect_list.end(), connection) == m_connect_list.end())
            {
                m_connect_list.push_back(connection);
            }
        }
    }

    return ZJV_STATUS_OK;
}


int Pipeline::parse_cfg_file(std::string cfg_file) 
{
    // 1. 解析配置文件，获取节点参数
    // Read the JSON file
    std::ifstream i(cfg_file);
    nlohmann::json j;
    i >> j;

    // Parse the JSON
    m_task_name = j["task_name"];
    m_expand_pipe = j["expand_pipe"].get<bool>();
    m_channel_num = j["channel_num"].get<int>();
    m_nodeparams = j["nodes"].get<std::vector<NodeParam>>();

    // 遍历所有节点，提取节点间的连接关系
    for(const auto & param : m_nodeparams)
    {
        for(const auto & input : param.m_input_node_datas)
        {
            m_connect_list.push_back({input.first, param.m_node_name});
        }
    }

    if(m_expand_pipe && (m_channel_num > 1) )
    {
        expand_pipe();
    }


    // ------------------打印信息------------------------------
    CLOG(INFO, PIPE_LOG) << "****************** PIPELINE INFO ******************** " ;
    CLOG(INFO, PIPE_LOG) << "m_task_name: " << m_task_name ;
    CLOG(INFO, PIPE_LOG) << "m_expand_pipe: " << m_expand_pipe ;
    CLOG(INFO, PIPE_LOG) << "m_channel_num: " << m_channel_num ;
    CLOG(INFO, PIPE_LOG) << "m_nodeparams.size(): " << m_nodeparams.size() ;
    for (auto & node : m_nodeparams) 
    {
        CLOG(INFO, PIPE_LOG) << "   node.m_node_type: " << node.m_node_type ;
        CLOG(INFO, PIPE_LOG) << "   node.m_node_name: " << node.m_node_name ;
        CLOG(INFO, PIPE_LOG) << "   node.m_channels: " << node.m_channels ;
        CLOG(INFO, PIPE_LOG) << "   node.m_cfg_file: " << node.m_cfg_file  ;
        CLOG(INFO, PIPE_LOG) << "   node.m_channel_id: " << node.m_channel_id  ;
        CLOG(INFO, PIPE_LOG) << "   ----------------------------------";
    }
    CLOG(INFO, PIPE_LOG) << "   m_connect_list.size(): " << m_connect_list.size() ;
    for (auto & item : m_connect_list) 
    {
        CLOG(INFO, PIPE_LOG) << "       " << item.first << " ===> " << item.second ;
    }
    CLOG(INFO, PIPE_LOG) ;


    return ZJV_STATUS_OK;
}


int Pipeline::init() 
{
    // 1. 初始化节点
    for (auto & node_param : m_nodeparams) 
    {
        std::shared_ptr<AbstractNode> node = NodeRegister::CreateNode(node_param);
        m_node_map.insert(std::make_pair(node_param.m_node_name, node));
    }

    //  2. 配置节点输入输出队列
    for (auto & connection : m_connect_list) 
    {
        std::string prior = connection.first;
        std::string next = connection.second;
        // 创建安全队列
        std::shared_ptr<FlowQueue> queue = std::make_shared<FlowQueue>();
        m_connectQueue.insert({prior+"-"+next, queue});

        m_node_map[prior]->connect_add_output(m_node_map[next]->get_name(), queue);
        m_node_map[next]->connect_add_input(m_node_map[prior]->get_name(), queue);
    }

    std::vector<std::string> zeroInDegreeNodes = getZeroInDegreeNodes(m_connect_list);
    for (auto & node_id : zeroInDegreeNodes) 
    {
        // 创建队列
        std::shared_ptr<FlowQueue> queue = std::make_shared<FlowQueue>();
        m_srcQueueList.insert(std::make_pair(m_node_map[node_id]->get_name(), queue));
        queue->set_buffer_strategy(BufferOverStrategy::ZJV_QUEUE_DROP_LATE);
        m_node_map[node_id]->connect_add_input(m_node_map[node_id]->get_name(), queue);
    }
    // 从m_connect_list 有向图连接中 提取末尾节点，即只有入度，没有出度
    std::vector<std::string> zeroOutDegreeNodes = getZeroOutDegreeNodes(m_connect_list);
    for (auto & node_id : zeroOutDegreeNodes) 
    {
        // 创建队列
        std::shared_ptr<FlowQueue> queue = std::make_shared<FlowQueue>();
        queue->set_buffer_strategy(BufferOverStrategy::ZJV_QUEUE_DROP_EARLY);
        queue->setCond(m_out_cond);
        m_dstQueueList.insert(std::make_pair(m_node_map[node_id]->get_name(), queue));
        m_node_map[node_id]->connect_add_output(m_node_map[node_id]->get_name(), queue);
    }
    // m_connectQueue
    CLOG(INFO, PIPE_LOG) << "m_connectQueue.size: " << m_connectQueue.size() ;
    for (auto & item : m_connectQueue) 
    {
        CLOG(INFO, PIPE_LOG) << "   m_connectQueue: " << item.first ;
    }

    // m_dstQueueList， m_srcQueueList， 的数量及其ID或名称
    CLOG(INFO, PIPE_LOG) << "m_srcQueueList.size(): " << m_srcQueueList.size() ;
    for (auto & item : m_srcQueueList) 
    {
        CLOG(INFO, PIPE_LOG) << "   m_srcQueueList: " << item.first ;
    }
    CLOG(INFO, PIPE_LOG) << "m_dstQueueList.size(): " << m_dstQueueList.size() ;
    for (auto & item : m_dstQueueList) 
    {
        CLOG(INFO, PIPE_LOG) << "   m_dstQueueList: " << item.first ;
    }

    CLOG(INFO, PIPE_LOG) << "Pipeline initialized" ;



    m_initialized = true;
    return ZJV_STATUS_OK;
}


std::vector<std::string> Pipeline::getZeroInDegreeNodes(const std::vector<std::pair<std::string,std::string>>& connect_list) 
{

    std::map<std::string, int> inDegree;

    // Initialize in-degrees
    for (const auto& pair : connect_list) {
        inDegree[pair.first] = 0;
        inDegree[pair.second] = 0;
    }

    // Calculate in-degrees
    for (const auto& pair : connect_list) {
        inDegree[pair.second]++;
    }

    // Find nodes with in-degree 0
    std::vector<std::string> zeroInDegreeNodes;
    for (const auto& pair : inDegree) {
        if (pair.second == 0) {
            zeroInDegreeNodes.push_back(pair.first);
        }
    }

    return zeroInDegreeNodes;
}



std::vector<std::string> Pipeline::getZeroOutDegreeNodes(const std::vector<std::pair<std::string,std::string>>& connect_list)
{
    std::map<std::string, int> outDegree;

    // Initialize out-degrees
    for (const auto& pair : connect_list) {
        outDegree[pair.first] = 0;
        outDegree[pair.second] = 0;
    }

    // Calculate out-degrees
    for (const auto& pair : connect_list) {
        outDegree[pair.first]++;
    }

    // Find nodes with out-degree 0
    std::vector<std::string> zeroOutDegreeNodes;
    for (const auto& pair : outDegree) {
        if (pair.second == 0) {
            zeroOutDegreeNodes.push_back(pair.first);
        }
    }

    return zeroOutDegreeNodes;
}

// 获取源节点
std::vector<std::string> Pipeline::get_src_node_name() 
{
    std::vector<std::string> src_node_name;
    for (auto & item : m_srcQueueList) 
    {
        src_node_name.push_back(item.first);
    }
    return src_node_name;
}

// 获取末尾节点
std::vector<std::string> Pipeline::get_dst_node_name() 
{
    std::vector<std::string> dst_node_name;
    for (auto & item : m_dstQueueList) 
    {
        dst_node_name.push_back(item.first);
    }
    return dst_node_name;
}

// 给源节点添加数据
int Pipeline::set_input_data(const std::string & tag, const std::shared_ptr<FlowData> & data) 
{
    if (m_srcQueueList.find(tag) != m_srcQueueList.end()) 
    {
        
        if(m_srcQueueList.size() > 1)
        {
            // 无法解析，采用默认管道编号（-1）
            // 解析后，设置flowdata管道编号
            int id = parse_id(tag);
            if (id >= 0 ) 
            {
                data->set_channel_id(id);
            }
            m_srcQueueList[tag]->Push(data);
            // CLOG(INFO, PIPE_LOG) << "set_input_data " << m_srcQueueList[tag]->size() ;
        }
        else
        {
            // 仅有单通道
            m_srcQueueList[tag]->Push(data);
        }

        return ZJV_STATUS_OK;
    }
    else
    {
        CLOG(INFO, PIPE_LOG) << "No such src node: " << tag ;
        return ZJV_STATUS_ERROR;
    }
}

// 从末尾节点提取数据
int Pipeline::get_output_data(const std::string & tag, std::shared_ptr<FlowData> & data) 
{
    if (m_dstQueueList.find(tag) != m_dstQueueList.end()) 
    {
        m_dstQueueList[tag]->Pop(data);
        return ZJV_STATUS_OK;
    }
    else
    {
        CLOG(INFO, PIPE_LOG) << "No such dst node: " << tag ;
        return ZJV_STATUS_ERROR;
    }
}


int Pipeline::show_debug_info()
{
    std::string str = "queue_size: ";
    for(const auto & buffer: m_srcQueueList)
    {
        str += buffer.first;
        str += "[";
        str += std::to_string(buffer.second->size());
        str += ",";
        str += std::to_string(buffer.second->get_drop_count());
        str +="] ";
    }
    str += "| ";

    for(const auto & buffer: m_dstQueueList)
    {
        str += buffer.first;
        str += "[";
        str += std::to_string(buffer.second->size());
        str += ",";
        str += std::to_string(buffer.second->get_drop_count());
        str +="] ";
    }
    str += "| ";
    CLOG(INFO, PIPE_LOG) << str ;
    str = "";
    
    for(const auto & buffer: m_connectQueue)
    {
        str += buffer.first;
        str += "[";
        str += std::to_string(buffer.second->size());
        str += ",";
        str += std::to_string(buffer.second->get_drop_count());
        str +="] ";
    }
    str += "| ";


    CLOG(INFO, PIPE_LOG) << str ;
    return ZJV_STATUS_OK;
}

}  // namespace ZJVIDEO
