#ifndef __ZJV_CLASSIFYPOSTPROCESS_H__
#define __ZJV_CLASSIFYPOSTPROCESS_H__

#include "nodes/infer_nodes/PostProcessor.h"

namespace ZJVIDEO
{

    enum ClassifyAlgorithm
    {
        ZJV_CLASSIFY_ALGORITHM_SOFTMAX = 0,
        ZJV_CLASSIFY_ALGORITHM_SIGMOID = 1,
        ZJV_CLASSIFY_ALGORITHM_MSE = 2,
    };

    class ClassifyPostProcessor:public PostProcessor
    {
    public:
        ClassifyPostProcessor();
        ~ClassifyPostProcessor() = default;

        virtual int parse_json(const nlohmann::json & j) override;
        virtual int run(std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois) override;
    private:
        int  m_num_classes;
        int  m_algorithm;
        float  m_attr_value_norm;
    };


} // namespace ZJV


#endif // __ZJV_CLASSIFYPOSTPROCESS_H__