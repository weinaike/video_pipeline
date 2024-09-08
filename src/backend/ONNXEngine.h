#ifndef _ZJV_ONNX_ENGINE_H_
#define _ZJV_ONNX_ENGINE_H_



#ifdef Enable_ONNX

#include <string>
#include <vector>
#include "EngineFactory.h"

#include "onnxruntime_cxx_api.h"

namespace ZJVIDEO{

// https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html


class ONNXEngine: public AbstractEngine{

public:
    ONNXEngine(const EngineParameter & param);
    ~ONNXEngine() override;

    virtual int init(const EngineParameter&) override;
    
    virtual int forward(std::vector<FBlob> &inputs, std::vector<FBlob> &outputs);
private:
    int forward_in(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape, 
        std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape);
    Ort::Env _env;
    Ort::SessionOptions _session_options;
    std::shared_ptr<Ort::Session> _session;
//        OrtMemoryInfo* _memory_info;
};


} //ZJVIDEO

#endif //Enable_ONNX    

#endif  // _ZJV_ONNX_ENGINE_H_
