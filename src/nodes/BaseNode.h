#ifndef ZJVIDEO_BASENODE_H
#define ZJVIDEO_BASENODE_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "../common/CommonDefine.h"
#include "../logger/easylogging++.h"

namespace ZJVIDEO {

// BaseNode基类职责：
// 1. 创建对象, 解析配置， 创建队列集合
// 2. pipeline调用connect,disconnect相关函数，配置流程关系。建立观察者模式
// 3. 启动进程，start(); 调用线程任务函数worker,执行任务流程
// 4. worker中包含核心处理流程， 保护获取数据， 处理数据，发送数据等
// 5. 结束线程

#define BASENODE_LOG "BaseNode"

class BaseNode : public AbstractNode {

public:

    BaseNode(const NodeParam & param);
    virtual ~BaseNode();
    BaseNode() = delete;

    //配置输入输出队列  
    virtual int connect_add_input(const std::string &, std::shared_ptr<ThreadSaveQueue> ) override ;
    virtual int connect_add_output(const std::string &, std::shared_ptr<ThreadSaveQueue> ) override;
    virtual int disconnect_del_input(const std::string &) override;
    virtual int disconnect_del_output(const std::string &) override;
    

    //启动进程，函数内部调用worker， int错误码
    virtual int start() override;  
    //停止进程
    virtual int stop() override;  

    virtual bool get_run_status() override;  
    virtual std::string get_name() override;

protected:
    // 主进程函数，没有必要为虚函数，调用流程基本固定  
    virtual int worker() ;
    //parse,解析配置文件
    virtual int parse_configure(std::string cfg_file); 
    //根据配置文件， 初始化对象,输入输出队列
    virtual int init(); 
    //实际主处理
    virtual int process(std::vector<std::shared_ptr<BaseData>> & data); 

protected:
    std::string                         m_log_name = "BaseNode";
    unsigned int                        m_node_id; // 节点id
    std::string                         m_node_type; // 节点类型
    std::string                         m_name;    // 节点名 
    std::thread                         m_worker; // 运行进程
    bool                                m_run = false; // 进程状态
    int                                 m_channels = 1; // 通道数
    int                                 m_get_data_max_num = 1;    //队列最大值
    std::mutex                          m_base_mutex;     // 操作队列是使用锁与条件
    // 条件变量
    std::shared_ptr<std::condition_variable> m_base_cond = std::make_shared<std::condition_variable>();

    std::map<std::string, std::shared_ptr<ThreadSaveQueue>> m_input_buffers; //输入队列集合
    std::map<std::string, std::shared_ptr<ThreadSaveQueue>> m_output_buffers;//输出队列集合
    std::string                         m_cfg_file; // 配置文件路径

};

}


#endif