#ifndef __ZJVIDEO_TRACKNODE_H
#define __ZJVIDEO_TRACKNODE_H

#include "../BaseNode.h"
#include "sort/tracker.h"
namespace ZJVIDEO
{

    class TrackNode : public BaseNode
    {

    public:
        TrackNode(const NodeParam &param);
        virtual ~TrackNode();
        TrackNode() = delete;

    protected:
        // parse,解析配置文件
        virtual int parse_configure(std::string cfg_file);
        // 根据配置文件， 初始化对象,输入输出队列
        virtual int init();

        virtual int process_single(const std::vector<std::shared_ptr<const BaseData>> &in_metas,
                                   std::vector<std::shared_ptr<BaseData>> &out_metas);


    protected:
        std::shared_ptr<Tracker>        m_tracker;

        int m_max_coast_cycles  = 1;
        int m_min_hits          = 3;
        // Set threshold to 0 to accept all detections
        float m_min_conf        = 0.6;

    }; // class TrackNode

}

#endif