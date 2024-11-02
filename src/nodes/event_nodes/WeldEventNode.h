#ifndef ZJVIDEO_WELDEVENTNODE_H
#define ZJVIDEO_WELDEVENTNODE_H

#include "nodes/BaseNode.h"

namespace ZJVIDEO {

class WeldEventNode : public BaseNode {

public:

    WeldEventNode(const NodeParam & param);
    virtual ~WeldEventNode();
    WeldEventNode() = delete;

protected:
    //parse,解析配置文件
    virtual int parse_configure(std::string cfg_file); 
    //根据配置文件， 初始化对象,输入输出队列
    virtual int init();         

    virtual int process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas);   
    virtual int control(std::shared_ptr<ControlData> &data) override;
private:
    std::deque<int> m_status;
    std::deque<float> m_score;
    std::deque<float> m_depth;
    std::deque<float> m_front;
    std::deque<float> m_back;
    int m_max_size = 8;
}; // class WeldEventNode

} // namespace ZJVIDEO


#endif