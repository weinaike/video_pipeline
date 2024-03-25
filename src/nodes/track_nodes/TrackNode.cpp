

#include "TrackNode.h"

#define TrkLOG "TrackNode"

namespace ZJVIDEO
{

    TrackNode::TrackNode(const NodeParam &param) : BaseNode(param)
    {
        el::Loggers::getLogger(TrkLOG);

        parse_configure(param.m_cfg_file);
        init();
        m_max_batch_size = 1;
        m_batch_process = false;

        CLOG(INFO, TrkLOG) << "TrackNode::TrackNode";
    }

    TrackNode::~TrackNode()
    {
        CLOG(INFO, TrkLOG) << "TrackNode::~TrackNode";
    }

    int TrackNode::parse_configure(std::string cfg_file)
    {
        CLOG(INFO, TrkLOG) << "TrackNode::parse_configure";
        std::ifstream i(cfg_file);
        if(i.is_open() == false)
        {
            CLOG(ERROR, TrkLOG) << "open cfg_file failed";
            m_max_coast_cycles  = 1;
            m_min_hits          = 3;
            // Set threshold to 0 to accept all detections
            m_min_conf          = 0.2;  // 最小置信度


            // 外部配置项
            m_main_category     = 0;    // 主类别
            m_sub_category      = 0;    // 子类别
            m_max_track_length  = 30;   // 跟踪的最大长度
            m_max_track_num     = 100;  // 跟踪的最大数量


            return ZJV_STATUS_ERROR;
        }
        nlohmann::json j;
        i >> j;
        m_main_category = j["main_category"];
        m_sub_category = j["sub_category"];
        m_max_track_num = j["max_track_num"];
        m_max_track_length = j["max_track_length"];
        m_min_conf = j["min_confidence"];

        // 打印配置参数
        CLOG(INFO, TrkLOG) << "----------------track node config-----------------";
        CLOG(INFO, TrkLOG) << "main_category:    [" << m_main_category << "]";
        CLOG(INFO, TrkLOG) << "sub_category:     [" << m_sub_category << "]";
        CLOG(INFO, TrkLOG) << "max_track_num:    [" << m_max_track_num << "]";
        CLOG(INFO, TrkLOG) << "max_track_length: [" << m_max_track_length << "]";
        CLOG(INFO, TrkLOG) << "min_confidence:   [" << m_min_conf << "]";
        CLOG(INFO, TrkLOG) << "--------------------------------------------------";    

        return 0;
    }

    int TrackNode::init()
    {
        CLOG(INFO, TrkLOG) << "TrackNode::init";
        m_tracker = std::make_shared<Tracker>(m_max_coast_cycles, m_min_conf);
        return 0;
    }

    int TrackNode::process_single(const std::vector<std::shared_ptr<const BaseData>> &in_metas,
                                  std::vector<std::shared_ptr<BaseData>> &out_metas)
    {
        std::shared_ptr<const FrameData> frame_data;
        std::vector<Rect> rois;
        for (int i = 0; i < in_metas.size(); i++)
        {
            if (in_metas[i]->data_name == "Frame")
            {
                frame_data = std::dynamic_pointer_cast<const FrameData>(in_metas[i]);
            }
            else if (in_metas[i]->data_name == "DetectResult")
            {
                std::shared_ptr<const DetectResultData> roi_data = std::dynamic_pointer_cast<const DetectResultData>(in_metas[i]);
                for (int j = 0; j < roi_data->detect_boxes.size(); j++)
                {
                    if(roi_data->detect_boxes[j].main_category != m_main_category || roi_data->detect_boxes[j].sub_category != m_sub_category)
                    {
                        continue;
                    }
                    Rect roi;
                    roi.x = roi_data->detect_boxes[j].x1;
                    roi.y = roi_data->detect_boxes[j].y1;
                    roi.width = (roi_data->detect_boxes[j].x2 - roi_data->detect_boxes[j].x1) / 2 * 2;
                    roi.height = (roi_data->detect_boxes[j].y2 - roi_data->detect_boxes[j].y1) / 2 * 2;
                    m_label = roi_data->detect_boxes[j].label;
                    rois.push_back(roi);
                }
            }
            else
            {
                CLOG(ERROR, TrkLOG) << "in_metas without Frame or ROI";
                assert(0);
            }
        }

        if(rois.size() == 0)
        {
            CLOG(INFO, TrkLOG) << "no roi to track";
            // std::shared_ptr<DetectResultData> out_data = std::make_shared<DetectResultData>();
            // out_data->data_type = ZJV_DATATYPE_DETECTRESULT_TRACK;
            // return ZJV_STATUS_OK;
        }

        m_tracker->Run(rois);
        std::map<int, Track>  tracks = m_tracker->GetTracks();
        

        // 生成输出数据
        std::shared_ptr<DetectResultData> out_data = std::make_shared<DetectResultData>();
        out_data->data_type = ZJV_DATATYPE_DETECTRESULT_TRACK;
        std::shared_ptr<DetectResultData> in_data = std::make_shared<DetectResultData>();
        for (auto &track : tracks)
        {
            int id = track.first;
            Track & trk =  track.second;
            Rect box = trk.GetStateAsBbox();
            DetectBox detect_box;
            detect_box.x1 = box.x;
            detect_box.y1 = box.y;
            detect_box.x2 = box.x + box.width;
            detect_box.y2 = box.y + box.height;
            detect_box.track_id = id;
            if(trk.coast_cycles_ == 0)
            {
                detect_box.track_status = ZJV_TRACK_STATUS_DETECTED;
            }
            else if(trk.coast_cycles_ > 0 && trk.coast_cycles_ < m_max_coast_cycles)
            {
                detect_box.track_status = ZJV_TRACK_STATUS_PREDICTED;
            }
            else
            {
                detect_box.track_status = ZJV_TRACK_STATUS_LOST;
            }
            if(trk.hit_streak_ < m_min_hits)
            {
                detect_box.track_status = ZJV_TRACK_STATUS_INIT;
            }
            detect_box.main_category = m_main_category;
            detect_box.sub_category = m_sub_category;
            detect_box.label = m_label;
            detect_box.score = 1.0;
            in_data->detect_boxes.push_back(detect_box); // 内部数据，每个目标不存储轨迹

            detect_box.track_boxes.push_back(box);
            // 轨迹回溯
            for(auto it = m_traj_data.rbegin(); it != m_traj_data.rend(); ++it)
            {          
                for(int j = 0; j < (*it)->detect_boxes.size(); j++)
                {
                    if((*it)->detect_boxes[j].track_id == id)
                    {
                        Rect r;
                        r.x = (*it)->detect_boxes[j].x1;
                        r.y = (*it)->detect_boxes[j].y1;
                        r.width = (*it)->detect_boxes[j].x2 - (*it)->detect_boxes[j].x1;
                        r.height = (*it)->detect_boxes[j].y2 - (*it)->detect_boxes[j].y1;
                        detect_box.track_boxes.push_back(r);
                    }
                }
            }            
            out_data->detect_boxes.push_back(detect_box);            
        }

        m_traj_data.push_back(in_data);
        if(m_traj_data.size() > m_max_track_length)
        {
            m_traj_data.pop_front();
        }
        out_metas.push_back(out_data);  

        return 0;
    }


REGISTER_NODE_CLASS(Track)

}   // namespace ZJVIDEO

