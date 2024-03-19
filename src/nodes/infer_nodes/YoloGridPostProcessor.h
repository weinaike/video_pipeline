#ifndef __ZJV_YOLOGRIDPOSTPROCESS_H__
#define __ZJV_YOLOGRIDPOSTPROCESS_H__

#include "PostProcessor.h"

namespace ZJVIDEO
{

    class YoloGridPostProcessor:public PostProcessor
    {
    public:
        YoloGridPostProcessor();
        ~YoloGridPostProcessor() = default;

        virtual int parse_configure(const std::string &cfg_file) override;
        virtual int run(const std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois) override;
    private:
        int             m_num_classes;
        float           m_conf_thres;
        float           m_iou_thres;
        std::string     m_output_data_type;
        std::string     m_post_type;

    };


} // namespace ZJV


#endif // __ZJV_POSTPROCESS_H__