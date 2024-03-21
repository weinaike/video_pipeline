#ifndef __ZJV_CLASSIFYPOSTPROCESS_H__
#define __ZJV_CLASSIFYPOSTPROCESS_H__

#include "../PostProcessor.h"

namespace ZJVIDEO
{

    class ClassifyPostProcessor:public PostProcessor
    {
    public:
        ClassifyPostProcessor();
        ~ClassifyPostProcessor() = default;

        virtual int parse_json(const nlohmann::json & j) override;
        virtual int run(std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois) override;
    private:
        int                         m_num_classes;
    };


} // namespace ZJV


#endif // __ZJV_CLASSIFYPOSTPROCESS_H__