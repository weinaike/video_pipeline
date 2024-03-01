
#ifndef _ZJ_VIDEO_PIPELINE_H
#define _ZJ_VIDEO_PIPELINE_H

#include <string>
#include <mutex>
#include <vector>
#include <atomic>
#include <memory>
#include <map>
#include "../common/CommonDefine.h"
#include "../logger/easylogging++.h"

#define PIPE_LOG "pipe"

namespace ZJVIDEO{

class Pipeline {

public:
    Pipeline() = default;

    explicit Pipeline(std::string cfg_file) ;

    virtual ~Pipeline();
    //init, start, stop
    int init();
    int start();
    int stop();


    // 给源节点添加数据
    std::vector<std::string> get_src_node_name();
    std::vector<std::string> get_dst_node_name();
    int set_input_data(const std::string & tag, std::shared_ptr<BaseData> data);
    // 从末尾节点提取数据
    int get_output_data(const std::string & tag, std::shared_ptr<BaseData> & data);

protected:
    
    //解析配置文件
    int parse_cfg_file(std::string cfg_file);
    std::vector<std::string> getZeroInDegreeNodes(const std::vector<std::pair<std::string ,std::string >>& connect_list);
    std::vector<std::string> getZeroOutDegreeNodes(const std::vector<std::pair<std::string ,std::string >>& connect_list);

    int expand_pipe();

protected:

    const std::string                                               m_log_name = "Pipeline";
    // 解析配置文件，存储
    std::string                                                     m_task_name;
    std::vector<NodeParam>                                          m_nodeparams;
    std::vector<std::pair<std::string, std::string> >               m_connect_list;  // 链接关系

    // 初始化变量
    std::map<std::string, std::shared_ptr<AbstractNode>>            m_node_map;    
    std::vector<std::shared_ptr<ThreadSaveQueue>>                   m_connectQueueList ;//每个连接的队列

    std::map<std::string, std::shared_ptr<ThreadSaveQueue>>         m_srcQueueList ;//每个连接的队列
    std::map<std::string, std::shared_ptr<ThreadSaveQueue>>         m_dstQueueList ;//每个连接的队列

    bool                                                            m_expand_pipe = false;
    int                                                             m_channel_num = 1;

    // 扩展后通道
    std::vector<NodeParam>                                          m_multi_channel_nodes;
    std::map<int, std::vector<NodeParam>>                           m_channels;
    std::map<int, std::vector<std::pair<std::string, std::string>>> m_channels_connect_list;

    std::mutex                                                      _mutex;
    std::atomic<bool>                                               m_initialized{false};

}; // class Pipeline


} // namespace ZJVIDEO

#endif  // _ZJ_VIDEO_PIPELINE_H
