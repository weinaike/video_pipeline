#include "ONNXEngine.h"
#include <numeric>
#include <assert.h>
#include "../logger/easylogging++.h"

namespace ZJVIDEO{

#define ONNX_LOG "ONNX"

ONNXEngine::ONNXEngine(const EngineParameter &param):AbstractEngine(param)
{
    el::Loggers::getLogger(ONNX_LOG);
    _session = nullptr;
    init(param);
}

ONNXEngine::~ONNXEngine()
{
    if (nullptr != _session) {
        _session->release();
        _session = nullptr;
    }
}

int ONNXEngine::forward(const void *frame, int frame_width, int frame_height, int frame_channel, 
                        std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape)
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> ort_inputs;

    std::vector<char*> input_node_names;
    for (auto & _input_node : m_input_nodes) 
    {
        std::vector<int64_t> input_dim(_input_node.second.begin(), _input_node.second.end());
        for(auto &dim: input_dim){
            if(dim == -1) dim = frame_height;
            if(dim == -2) dim = frame_width;
        }
        Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info,
                                                            (void *) frame,
                                                            frame_width * frame_height * frame_channel * sizeof(float) ,
                                                            input_dim.data(),
                                                            input_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        ort_inputs.push_back(std::move(input_tensor));
        input_node_names.push_back(const_cast<char*>(_input_node.first.c_str()));
    }
    std::vector<Ort::Value> ort_outputs;
    std::vector<char*> output_node_names;

    for(const auto& name: m_output_node_name){
        output_node_names.push_back(const_cast<char*>(name.c_str()));

    }
    ort_outputs = _session->Run(Ort::RunOptions{nullptr},
                                    input_node_names.data(),
                                    ort_inputs.data(),
                                    ort_inputs.size(),
                                    output_node_names.data(),
                                    output_node_names.size());
    outputs.clear();
    outputs.resize(m_output_node_name.size());
    outputs_shape.clear();
    outputs_shape.resize(m_output_node_name.size());

    for (auto &ort_output: ort_outputs)
    {
        auto index = &ort_output - &ort_outputs[0];
        auto info = ort_output.GetTensorTypeAndShapeInfo();
        auto output_len = info.GetElementCount();
        auto dim_count = info.GetDimensionsCount();
        std::vector<int64_t> dims(dim_count, 1);
        info.GetDimensions(dims.data(), info.GetDimensionsCount());

        for (int cc = 0; cc < dim_count; cc++){
            outputs_shape[index].push_back(int(dims[cc]));
        }
        ort_output.GetTensorData<float>();
        outputs[index].resize(output_len);
        ::memcpy(outputs[index].data(), ort_output.GetTensorData<float>(), sizeof(float)*output_len);

    }
    return ZJV_STATUS_OK;
}

int ONNXEngine::init(const EngineParameter &param)
{
    m_model_name = param.m_model_name;
    m_backend_name = param.m_engine_type;
    m_input_node_name.assign(param.m_input_node_name.begin(), param.m_input_node_name.end());
    m_output_node_name.assign(param.m_output_node_name.begin(), param.m_output_node_name.end());
    m_input_nodes = param.m_input_nodes;

    _env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, m_model_name.c_str());

    // cuda
    // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);

    _session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    _session_options.SetLogSeverityLevel(ORT_LOGGING_LEVEL_WARNING);

    _session = std::make_shared<Ort::Session>(_env, param.m_model_path.c_str(), _session_options);

    if (nullptr == _session) {
        CLOG(ERROR, ONNX_LOG)<<"backend onnx init failed!";
        return ZJV_STATUS_ERROR;
    }
    CLOG(INFO, ONNX_LOG)<<"backend onnx init succeed!";
    return ZJV_STATUS_OK;
}

int ONNXEngine::init(const EngineParameter &, const void *buffer_in1, const void *buffer_in2) {
    return ZJV_STATUS_NOT_IMPLEMENT;
}

int ONNXEngine::forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape,
                            std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) {

    assert(m_input_nodes.size() == input.size());
    assert(input_shape.size() == input.size());

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    std::vector<Ort::Value> ort_inputs;

    std::vector<char*> input_node_names;
    for(int i = 0; i < m_input_node_name.size(); i++)
    {
        auto _input_node = m_input_nodes[m_input_node_name[i]];
        std::vector<int64_t> input_dim(_input_node.begin(), _input_node.end());
        for(int d = 0; d < input_dim.size(); d++)
        {
            if(input_dim[d] < 0){
                input_dim[d] = input_shape[i][d];
            }
        }

        Ort::Value input_tensor = Ort::Value::CreateTensor(memory_info,
                                                            (void *) input[i],
                                                            std::accumulate(
                                                                    input_shape[i].begin(),
                                                                    input_shape[i].end(),
                                                                    1,
                                                                    std::multiplies<int>()
                                                                    ) * sizeof(float),
                                                            input_dim.data(),
                                                            input_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
        ort_inputs.push_back(std::move(input_tensor));
        input_node_names.push_back(const_cast<char*>(m_input_node_name[i].c_str()));
    }

    std::vector<Ort::Value> ort_outputs;
    std::vector<char*> output_node_names;

    for(const auto& name: m_output_node_name){
        output_node_names.push_back(const_cast<char*>(name.c_str()));

    }

    ort_outputs = _session->Run(Ort::RunOptions{nullptr},
                                input_node_names.data(),
                                ort_inputs.data(),
                                ort_inputs.size(),
                                output_node_names.data(),
                                output_node_names.size());

    outputs.clear();
    outputs.resize(m_output_node_name.size());
    outputs_shape.clear();
    outputs_shape.resize(m_output_node_name.size());

    for (auto &ort_output: ort_outputs){
        auto index = &ort_output - &ort_outputs[0];
        auto info = ort_output.GetTensorTypeAndShapeInfo();
        auto output_len = info.GetElementCount();
        auto dim_count = info.GetDimensionsCount();
        std::vector<int64_t> dims(dim_count, 1);
        info.GetDimensions(dims.data(), info.GetDimensionsCount());

        for (int cc = 0; cc < dim_count; cc++){
            outputs_shape[index].push_back(int(dims[cc]));
        }

        ort_output.GetTensorData<float>();
        outputs[index].resize(output_len);
        ::memcpy(outputs[index].data(), ort_output.GetTensorData<float>(), sizeof(float)*output_len);

    }
    // std::cout<<"onnx"<<std::endl;
    return ZJV_STATUS_OK;
}



REGISTER_ENGINE_CLASS(ONNX)
} // namespace ZJVIDEO