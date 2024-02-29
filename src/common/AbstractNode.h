
#ifndef ZJVIDEO_ABSTRACTNODE_H
#define ZJVIDEO_ABSTRACTNODE_H

#include <string>
#include <memory>
#include <vector>
#include "NodeParam.h"
#include "ThreadSaveQueue.h"
#include "StatusCode.h"


namespace ZJVIDEO {

class AbstractNode {

public:
    // 1. 通过工厂方法创建实例
    AbstractNode(const NodeParam & param){};

    AbstractNode() = delete;
    // 2. 配置节点输入输出队列
    virtual int connect_add_input(const std::string &, std::shared_ptr<ThreadSaveQueue> ) = 0;
    virtual int connect_add_output(const std::string &, std::shared_ptr<ThreadSaveQueue> ) = 0;
    virtual int disconnect_del_input(const std::string &) = 0;
    virtual int disconnect_del_output(const std::string &) = 0;

    // 3. 启动节点
    virtual int start() = 0; 
    // 4. 停止节点
    virtual int stop() = 0;  
    // 查询状态
    virtual bool get_run_status() = 0;  
    virtual std::string get_name() = 0;  
    // 5. 删除节点
    virtual ~AbstractNode() = default;
  
}; // class AbstractNode

}  // namespace ZJVIDEO

#endif  // ZJVIDEO_ABSTRACTNODE_H