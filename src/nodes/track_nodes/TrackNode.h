#ifndef __ZJVIDEO_TRACKNODE_H
#define __ZJVIDEO_TRACKNODE_H

/**
 * 跟踪算法的配置项：
 * 基本不需要外部设置项
 * 
 * 
 * 以下都是内部配置项
 * 1. max_coast_cycles: 没有检测结构，纯预测的最大次数，超过这个次数就认为丢失
 * 2. min_hits: 命中的累计次数，至少累计多少次才能被认为是跟踪到
 * 3. min_conf: 最小的置信度
 * 
 *  
 * 关联跟踪，同一ID包含不同类别标签，这个比较麻烦
 * 
 * 单类别跟踪，如何筛选类别
 * 
 * 输出的数据结构
 * 1. ID
 * 2. 主类别
 * 3. 次类别
 * 4. 跟踪框
 * 5. 跟踪置信度
 * 6. 关联类别框
 * 7. 跟踪框的状态，（丢失，检测匹配，预测，初始化）
 * 
 * 按类别跟踪，同一类别内匹配
 * 
 * 1. 所有类别都跟踪
 * 2. 筛选类别跟踪
 *  
 * 
 * 
 **/






#include "../BaseNode.h"
#include "sort/tracker.h"
#include <deque>
#include "common/Shape.h"
#include "nlohmann/json.hpp"

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
        float m_min_conf        = 0.2;  // 最小置信度


        // 外部配置项
        int m_main_category         = 0;    // 主类别
        int m_sub_category          = 0;    // 子类别
        int m_max_track_length      = 30;   // 跟踪的最大长度
        int m_max_track_num         = 100;  // 跟踪的最大数量
        int m_label = 0;
        
        std::deque<std::shared_ptr<DetectResultData>> m_traj_data;
        
    }; // class TrackNode

}

#endif