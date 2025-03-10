#ifndef ZJVIDEO_IMAGESRCNODE_H
#define ZJVIDEO_IMAGESRCNODE_H

#include "nodes/BaseNode.h"

namespace ZJVIDEO {

// BaseNode基类职责：
// 1. 创建对象, 解析配置， 创建队列集合
// 2. pipeline调用connect,disconnect相关函数，配置流程关系。建立观察者模式
// 3. 启动进程，start(); 调用线程任务函数worker,执行任务流程
// 4. worker中包含核心处理流程， 保护获取数据， 处理数据，发送数据等
// 5. 结束线程

class ImageSrcNode : public BaseNode {

public:

    ImageSrcNode(const NodeParam & param);
    virtual ~ImageSrcNode();
    ImageSrcNode() = delete;

protected:
    //parse,解析配置文件
    virtual int parse_configure(std::string cfg_file); 
    //根据配置文件， 初始化对象,输入输出队列
    virtual int init();         

    virtual int process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas);

protected:


}; // class ImageSrcNode

} // namespace ZJVIDEO


#endif