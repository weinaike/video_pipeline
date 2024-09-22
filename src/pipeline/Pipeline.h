
#ifndef _ZJ_VIDEO_PIPELINE_H
#define _ZJ_VIDEO_PIPELINE_H

#define ELPP_THREAD_SAFE
// #define ELPP_EXPERIMENTAL_ASYNC
#include "logger/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

#include <string>
#include <mutex>
#include <vector>
#include <atomic>
#include <memory>
#include <map>
#include "common/CommonDefine.h"
#include "public/PublicPipeline.h"


#define PIPE_LOG "pipe"

namespace ZJVIDEO{

class Pipeline : public PublicPipeline {

public:
    explicit Pipeline(std::string cfg_file) ;

    virtual ~Pipeline();
    //init, start, stop
    int init() override;
    int start() override;
    int stop() override;


    // 给源节点添加数据
    int set_input_data(const std::shared_ptr<FrameData> & data) override;
    int set_input_data(const std::shared_ptr<VideoData> & data) ;
    // 从末尾节点提取数据
    int get_output_data(std::vector<std::shared_ptr<EventData> >  & data) override;
    int get_output_data(std::vector<std::shared_ptr<const BaseData>> & data) ;

    // 设置配置参数
    int control(std::shared_ptr<ControlData>& data ) override;
    int show_debug_info() override;


protected:
    
    //解析配置文件
    int parse_cfg_file(std::string cfg_file);
    std::vector<std::string> getZeroInDegreeNodes(const std::vector<std::pair<std::string ,std::string >>& connect_list);
    std::vector<std::string> getZeroOutDegreeNodes(const std::vector<std::pair<std::string ,std::string >>& connect_list);

    int expand_pipe();
    std::vector<std::string> get_src_node_name();
    std::vector<std::string> get_dst_node_name();

protected:
    // 解析配置文件，存储
    std::string                                                     m_task_name;
    std::vector<NodeParam>                                          m_nodeparams;
    std::vector<std::pair<std::string, std::string> >               m_connect_list;  // 链接关系

    // 初始化变量
    std::map<std::string, std::shared_ptr<AbstractNode>>            m_node_map;    
    std::map<std::string, std::shared_ptr<FlowQueue>>               m_connectQueue ;//每个连接的队列

    std::map<std::string, std::shared_ptr<FlowQueue>>               m_srcQueueList ;//每个连接的队列
    std::map<std::string, std::shared_ptr<FlowQueue>>               m_dstQueueList ;//每个连接的队列
    std::map<int, std::string>                                      m_src_map ;//
    std::map<int, std::string>                                      m_dst_map ;//
    std::vector<std::string>                                        m_src_node_name;
    std::vector<std::string>                                        m_dst_node_name;

    bool                                                            m_expand_pipe = false;
    int                                                             m_channel_num = 1;
    int                                                             m_queue_size = 32;

    // 扩展后通道
    std::vector<NodeParam>                                          m_multi_channel_nodes;
    std::map<int, std::vector<NodeParam>>                           m_channels;
    std::map<int, std::vector<std::pair<std::string, std::string>>> m_channels_connect_list;

    std::mutex                                                      m_mutex;
    std::shared_ptr<std::condition_variable> m_out_cond = std::make_shared<std::condition_variable>();
    std::atomic<bool>                                               m_initialized{false};
    el::Logger*                                                     m_logger;
}; // class Pipeline


} // namespace ZJVIDEO

#endif  // _ZJ_VIDEO_PIPELINE_H
