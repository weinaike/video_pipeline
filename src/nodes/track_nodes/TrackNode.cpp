

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
        return 0;
    }

    int TrackNode::init()
    {
        CLOG(INFO, TrkLOG) << "TrackNode::init";
        m_tracker = std::make_shared<Tracker>();
        return 0;
    }

    int TrackNode::process_single(const std::vector<std::shared_ptr<const BaseData>> &in_metas,
                                  std::vector<std::shared_ptr<BaseData>> &out_metas)
    {
        CLOG(INFO, TrkLOG) << "TrackNode::process_single";

        if (in_metas.size() < 2)
        {
            CLOG(ERROR, TrkLOG) << "in_metas size < 2";
            return ZJV_STATUS_ERROR;
        }

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
                    Rect roi;
                    roi.x = roi_data->detect_boxes[j].x1;
                    roi.y = roi_data->detect_boxes[j].y1;
                    roi.width = (roi_data->detect_boxes[j].x2 - roi_data->detect_boxes[j].x1) / 2 * 2;
                    roi.height = (roi_data->detect_boxes[j].y2 - roi_data->detect_boxes[j].y1) / 2 * 2;
                    rois.push_back(roi);
                }
            }
            else
            {
                CLOG(ERROR, TrkLOG) << "in_metas without Frame or ROI";
                assert(0);
            }
        }

        return 0;
    }

}
