#ifndef __ZJV_PREPROCESS_H__
#define __ZJV_PREPROCESS_H__

#include <string>
#include <vector>
#include <memory>
#include "InferDefine.h"

namespace ZJVIDEO
{
    class PreProcessor
    {
    public:
        PreProcessor(int lib_type);
        ~PreProcessor() = default;

        int run(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param);
        int parse_json(const nlohmann::json & j);
        PreProcessParameter get_param() { return m_param; }
    protected:
        int run_cimg(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param);
        int run_cuda(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param);
    private:
        int m_lib_type;
        PreProcessParameter m_param;
    };

} // namespace ZJVIDEO

#endif // __ZJV_PREPROCESS_H__