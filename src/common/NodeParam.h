



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
    unsigned int                m_node_id;              // 节点id
    std::string                 m_node_type;            // 节点类型
    std::string                 m_node_name;            // 节点名
    int                         m_channels = 1;         // 通道数   
    std::string                 m_cfg_file;             // 配置文件路径
};// struct NodeParam



}  // namespace ZJVIDEO

#endif  // ZJVIDEO_ABSTRACTNODE_H