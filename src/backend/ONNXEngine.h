#ifndef _ZJV_ONNX_ENGINE_H_
#define _ZJV_ONNX_ENGINE_H_

#include <string>
#include <vector>
#include "EngineFactory.h"

#define Enable_ONNX

#ifdef Enable_ONNX

#include "OnnxRuntime/onnxruntime_cxx_api.h"

namespace ZJVIDEO{

// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html


class ONNXEngine: public AbstractEngine{

public:
    ONNXEngine(const EngineParameter & param);
    int init(const EngineParameter&) override;
    int init(const EngineParameter&, const void *buffer_in1, const void* buffer_in2) override;

    ~ONNXEngine() override;
    int forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
    int forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
private:
    Ort::Env _env;
    Ort::SessionOptions _session_options;
    std::shared_ptr<Ort::Session> _session;
//        OrtMemoryInfo* _memory_info;
};


} //ZJVIDEO

#endif //Enable_ONNX    

#endif  // _ZJV_ONNX_ENGINE_H_
