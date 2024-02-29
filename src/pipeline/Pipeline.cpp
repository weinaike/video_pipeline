//
// Created by lijin on 2023/12/6.
//

#include "Pipeline.h"
#include <../nlohmann/json.hpp>
#include <fstream>


namespace ZJVIDEO {


Pipeline::Pipeline(std::string cfg_file) {
    parse_cfg_file(cfg_file);
    m_initialized = false;
}


int Pipeline::start() 
{
    if (!m_initialized) 
    {
        std::cout << "Pipeline not initialized" << std::endl;
        return ZJV_STATUS_ERROR;
    }

    for (auto & node : m_node_map) 
    {
        node.second->start();
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
        std::cout << "Pipeline not initialized" << std::endl;
    }

    return ZJV_STATUS_OK;
}

Pipeline::~Pipeline() {
    stop();
}


static void from_json(const nlohmann::json& j, NodeParam& p) 
{
    j.at("node_id").get_to(p.m_node_id);
    j.at("node_type").get_to(p.m_node_type);
    j.at("node_name").get_to(p.m_node_name);
    j.at("channels").get_to(p.m_channels);
    j.at("cfg_file").get_to(p.m_cfg_file);
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
    m_nodeparams = j["nodes"].get<std::vector<NodeParam>>();

    for (const auto& connection : j["connections"]) {
        m_connect_list.push_back({connection["from_node_id"], connection["to_node_id"]});
    }




    // 打印信息
    std::cout << "m_task_name: " << m_task_name << std::endl;
    std::cout << "m_nodeparams.size(): " << m_nodeparams.size() << std::endl;
    for (auto & node : m_nodeparams) 
    {
        std::cout << "node.m_node_id: " << node.m_node_id << std::endl;
        std::cout << "node.m_node_type: " << node.m_node_type << std::endl;
        std::cout << "node.m_node_name: " << node.m_node_name << std::endl;
        std::cout << "node.m_channels: " << node.m_channels << std::endl;
        std::cout << "node.m_cfg_file: " << node.m_cfg_file << std::endl << std::endl;
    }
    std::cout << "m_connect_list.size(): " << m_connect_list.size() << std::endl;
    for (auto & item : m_connect_list) 
    {
        std::cout << "m_connect_list: " << item.first << " " << item.second << std::endl;
    }
    std::cout << std::endl;


    return ZJV_STATUS_OK;
}


int Pipeline::init() 
{
    // 1. 初始化节点
    for (auto & node_param : m_nodeparams) 
    {
        std::shared_ptr<AbstractNode> node = NodeRegister::CreateNode(node_param);
        m_node_map.insert(std::make_pair(node_param.m_node_id, node));
    }

    //  2. 配置节点输入输出队列
    for (auto & connection : m_connect_list) 
    {
        int prior = connection.first;
        int next = connection.second;
        // 创建安全队列
        std::shared_ptr<ThreadSaveQueue> queue = std::make_shared<ThreadSaveQueue>();
        m_connectQueueList.push_back(queue);

        m_node_map[prior]->connect_add_output(m_node_map[next]->get_name(), queue);
        m_node_map[next]->connect_add_input(m_node_map[prior]->get_name(), queue);
    }

    std::vector<int> zeroInDegreeNodes = getZeroInDegreeNodes(m_connect_list);
    for (auto & node_id : zeroInDegreeNodes) 
    {
        // 创建队列
        std::shared_ptr<ThreadSaveQueue> queue = std::make_shared<ThreadSaveQueue>();
        m_srcQueueList.insert(std::make_pair(m_node_map[node_id]->get_name(), queue));
        m_node_map[node_id]->connect_add_input(m_node_map[node_id]->get_name(), queue);
    }
    // 从m_connect_list 有向图连接中 提取末尾节点，即只有入度，没有出度
    std::vector<int> zeroOutDegreeNodes = getZeroOutDegreeNodes(m_connect_list);
    for (auto & node_id : zeroOutDegreeNodes) 
    {
        // 创建队列
        std::shared_ptr<ThreadSaveQueue> queue = std::make_shared<ThreadSaveQueue>();
        m_dstQueueList.insert(std::make_pair(m_node_map[node_id]->get_name(), queue));
        m_node_map[node_id]->connect_add_output(m_node_map[node_id]->get_name(), queue);
    }

    // m_dstQueueList， m_srcQueueList， 的数量及其ID或名称
    std::cout << "m_srcQueueList.size(): " << m_srcQueueList.size() << std::endl;
    for (auto & item : m_srcQueueList) 
    {
        std::cout << "m_srcQueueList: " << item.first << std::endl;
    }
    std::cout << "m_dstQueueList.size(): " << m_dstQueueList.size() << std::endl;
    for (auto & item : m_dstQueueList) 
    {
        std::cout << "m_dstQueueList: " << item.first << std::endl;
    }

    std::cout << "Pipeline initialized" << std::endl;



    m_initialized = true;
    return ZJV_STATUS_OK;
}


std::vector<int> Pipeline::getZeroInDegreeNodes(const std::vector<std::pair<int,int>>& connect_list) 
{

    std::map<int, int> inDegree;

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
    std::vector<int> zeroInDegreeNodes;
    for (const auto& pair : inDegree) {
        if (pair.second == 0) {
            zeroInDegreeNodes.push_back(pair.first);
        }
    }

    return zeroInDegreeNodes;
}



std::vector<int> Pipeline::getZeroOutDegreeNodes(const std::vector<std::pair<int,int>>& connect_list)
{
    std::map<int, int> outDegree;

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
    std::vector<int> zeroOutDegreeNodes;
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
int Pipeline::set_input_data(const std::string & tag, std::shared_ptr<BaseData> data) 
{
    if (m_srcQueueList.find(tag) != m_srcQueueList.end()) 
    {
        m_srcQueueList[tag]->Push(data);
        return ZJV_STATUS_OK;
    }
    else
    {
        std::cout << "No such src node: " << tag << std::endl;
        return ZJV_STATUS_ERROR;
    }
}

// 从末尾节点提取数据
int Pipeline::get_output_data(const std::string & tag, std::shared_ptr<BaseData> & data) 
{
    if (m_dstQueueList.find(tag) != m_dstQueueList.end()) 
    {
        m_dstQueueList[tag]->Pop(data);
        return ZJV_STATUS_OK;
    }
    else
    {
        std::cout << "No such dst node: " << tag << std::endl;
        return ZJV_STATUS_ERROR;
    }
}

}  // namespace ZJVIDEO
