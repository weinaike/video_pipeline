#ifndef ZJVIDEO_IMAGECACHENODE_H
#define ZJVIDEO_IMAGECACHENODE_H

#include "nodes/BaseNode.h"
#include "nodes/infer_nodes/InferDefine.h"
#include "common/Shape.h"

namespace ZJVIDEO {

// BaseNode基类职责：
// 1. 创建对象, 解析配置， 创建队列集合
// 2. pipeline调用connect,disconnect相关函数，配置流程关系。建立观察者模式
// 3. 启动进程，start(); 调用线程任务函数worker,执行任务流程
// 4. worker中包含核心处理流程， 保护获取数据， 处理数据，发送数据等
// 5. 结束线程
enum output_stream_type {
    ZJV_OUTPUT_STREAM_TYPE_CONTINUE = 0,
    ZJV_OUTPUT_STREAM_TYPE_TRIGGER = 1,    
};

class CacheNode : public BaseNode {

public:

    CacheNode(const NodeParam & param);
    virtual ~CacheNode();
    CacheNode() = delete;

protected:
    //parse,解析配置文件
    virtual int parse_configure(std::string cfg_file); 
    //根据配置文件， 初始化对象,输入输出队列
    virtual int init();         

    virtual int process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas);
    virtual int control(std::shared_ptr<ControlData> &data) override;
protected:
    int transfer_data(std::shared_ptr<const FrameData> in_frame_data, std::shared_ptr<FrameData> & out_frame_data);
    

protected:
    int m_count;                // 满足条件帧累计，满足帧频率条件，输出数据并清零
    int m_append_count;         // 未满足条件的帧累计，达到记录条件清零
    float * m_test_data = NULL;
    int m_test_data_size = 0;

private:
    std::list<std::shared_ptr<FrameData> > m_frame_datas;
    int m_os_type;     // output_stream_type  0: continue, 1: trigger
    int m_frame_num;    // 单次输出帧数量
    int m_step;         // 帧采样间隔
    float m_fps;        // 单次输出的最小频率

    bool m_transform;   // 是否需要转换
    int m_device_id;      // 设备id

    BaseDataType m_output_type;

    // 预处理参数
    PreProcessParameter m_param;
    Rect2f m_roi;


}; // class CacheNode

} // namespace ZJVIDEO


#endif