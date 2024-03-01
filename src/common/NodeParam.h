



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
};// struct NodeParam



}  // namespace ZJVIDEO

#endif  // ZJVIDEO_ABSTRACTNODE_H