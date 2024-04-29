#ifndef __ZJV_PREPROCESS_H__
#define __ZJV_PREPROCESS_H__

#include <string>
#include <vector>
#include <memory>
#include "InferDefine.h"
#include "backend/AbstractEngine.h"
#include "logger/easylogging++.h"
#define PRELOG "PreProc"

namespace ZJVIDEO
{
    class PreProcessor
    {
    public:
        PreProcessor(int lib_type, int device_id = 0);
        ~PreProcessor() = default;

        int run(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param);
        int parse_json(const nlohmann::json & j);
        PreProcessParameter get_param() { return m_param; }
        void set_engine(const std::shared_ptr<AbstractEngine>& eng){m_engine = eng;};
    protected:
        int run_cimg(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param);
        int run_cuda(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param);
        int run_3d_cuda(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param);
    private:
        int m_lib_type;
        PreProcessParameter m_param;
        int m_device_id;
        std::shared_ptr<AbstractEngine> m_engine;
    };

} // namespace ZJVIDEO

#endif // __ZJV_PREPROCESS_H__