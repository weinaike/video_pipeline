#ifndef ZJVIDEO_BASENODE_H
#define ZJVIDEO_BASENODE_H
#include <deque>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "common/CommonDefine.h"
#include "logger/easylogging++.h"

namespace ZJVIDEO
{

    // BaseNode基类职责：
    // 1. 创建对象, 解析配置， 创建队列集合
    // 2. pipeline调用connect,disconnect相关函数，配置流程关系。建立观察者模式
    // 3. 启动进程，start(); 调用线程任务函数worker,执行任务流程
    // 4. worker中包含核心处理流程， 保护获取数据， 处理数据，发送数据等
    // 5. 结束线程

    enum NODE_POSITION_TYPE
    {
        ZJV_NODE_POSITION_UNKNOWN = 0,
        ZJV_NODE_POSITION_SRC,
        ZJV_NODE_POSITION_MID,
        ZJV_NODE_POSITION_DST
    };

    class BaseNode : public AbstractNode
    {

    public:
        BaseNode(const NodeParam &param);
        virtual ~BaseNode();
        BaseNode() = delete;

        // 配置输入输出队列
        virtual int connect_add_input(const std::string &, std::shared_ptr<FlowQueue>) override;
        virtual int connect_add_output(const std::string &, std::shared_ptr<FlowQueue>) override;
        virtual int disconnect_del_input(const std::string &) override;
        virtual int disconnect_del_output(const std::string &) override;

        // 启动进程，函数内部调用worker， int错误码
        virtual int start() override;
        // 停止进程
        virtual int stop() override;

        virtual bool get_run_status() override;
        virtual int control(std::shared_ptr<ControlData> &data) override;
        virtual std::string get_name() override;

    protected:
        // 主进程函数，没有必要为虚函数，调用流程基本固定
        virtual int worker();
        // parse,解析配置文件
        virtual int parse_configure(std::string cfg_file);
        // 根据配置文件， 初始化对象,输入输出队列
        virtual int init();

        virtual int process_batch(const std::vector<std::vector<std::shared_ptr<const BaseData>>> &in_metas_batch,
                                  std::vector<std::vector<std::shared_ptr<BaseData>>> &out_metas_batch);

        virtual int process_single(const std::vector<std::shared_ptr<const BaseData>> &in_metas,
                                   std::vector<std::shared_ptr<BaseData>> &out_metas);

        // 实际主处理, 不能更改指针指向的对象， 但可以修改添加对象属性
        virtual int process(const std::vector<std::shared_ptr<FlowData>> &datas);

        virtual int get_input_data(std::vector<std::shared_ptr<FlowData>> &data);
        virtual int send_output_data(const std::vector<std::shared_ptr<FlowData>> &data);

    protected:
        NodeParam m_nodeparam;                                            // 节点参数
        bool m_batch_process = false;                                     // 是否需要批处理
        std::vector<std::string> m_input_data_names;                      // 所需的输入数据
        std::map<int, std::vector<std::string>> m_input_data_names_batch; //  多通道输入数据

        std::thread m_worker;     // 运行进程
        bool m_run = false;       // 进程状态
        int m_max_batch_size = 1; // 批处理最大数据量
        std::mutex m_base_mutex;  // 操作队列是使用锁与条件

        // 条件变量
        std::shared_ptr<std::condition_variable> m_base_cond = std::make_shared<std::condition_variable>();

        std::map<std::string, std::shared_ptr<FlowQueue>> m_input_buffers;  // 输入队列集合
        std::map<std::string, std::shared_ptr<FlowQueue>> m_output_buffers; // 输出队列集合
        std::string m_cfg_file;                                             // 配置文件路径
        int m_node_position_type;

        std::deque<std::string> m_dealed_smaple; // 历史处理样本
        float m_fps = 0;
        int m_fps_count = 0;
        el::Logger *m_logger;
    };

}

#endif