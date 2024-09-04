#ifndef __ZJV_YOLOGRIDPOSTPROCESS_H__
#define __ZJV_YOLOGRIDPOSTPROCESS_H__

#include "nodes/infer_nodes/PostProcessor.h"

namespace ZJVIDEO
{

    class YoloGridPostProcessor:public PostProcessor
    {
    public:
        YoloGridPostProcessor();
        ~YoloGridPostProcessor() = default;

        virtual int parse_json(const nlohmann::json & j) override;
        virtual int run(std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois) override;
    private:
        int                         m_num_classes;
        float                       m_conf_thres;
        float                       m_iou_thres;
    };


} // namespace ZJV


#endif // __ZJV_POSTPROCESS_H__