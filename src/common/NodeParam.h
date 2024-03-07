



#ifndef ZJVIDEO_NODEPARAM_H
#define ZJVIDEO_NODEPARAM_H

#include <string>
#include <memory>
#include <vector>

#include "ThreadSaveQueue.h"
#include "StatusCode.h"

namespace ZJVIDEO {

struct NodeParam
{
    std::string                 m_node_type;            // 节点类型
    std::string                 m_node_name;            // 节点名
    bool                        m_channels = true;      // 是否支持多通道
    std::string                 m_cfg_file;             // 配置文件路径
    int                         m_channel_id = -1;      // 通道id
    std::vector<std::string>    m_output_datas;         // 输出数据类型
    std::vector<std::pair<std::string, std::string>>    m_input_node_datas;   // 前置节点数据
    bool                        m_wait_data;
};// struct NodeParam



}  // namespace ZJVIDEO

#endif  // ZJVIDEO_ABSTRACTNODE_H