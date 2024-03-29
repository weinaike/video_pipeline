#include "TRTEngine.h"
#include "logger/easylogging++.h"
#ifdef Enable_TRT

#define TRT_LOG "TRT"

#include "TensorRT/NvOnnxParser.h"
#include <fstream>
#include <sstream>

#define CHECK_PTR(ptr)              \
if (ptr == nullptr)                 \
{                                   \
    CLOG(ERROR, TRT_LOG) << "ptr is nullptr!"; \
    return ZJV_STATUS_ERROR;        \
}

namespace ZJVIDEO
{
    // 获取文件后缀名
    std::string get_file_extension(const std::string& filename) 
    {
        size_t pos = filename.rfind('.');
        if (pos == std::string::npos) {
            // No extension found
            return "";
        } else {
            return filename.substr(pos);
        }
    }

    struct InferDeleter
    {
        template <typename T>
        void operator()(T* obj) const
        {
            delete obj;
        }
    };

    template <typename T>
    using UniquePtr = std::unique_ptr<T, InferDeleter>;



    static auto StreamDeleter = [](cudaStream_t* pStream)
    {
        if (pStream)
        {
            cudaStreamDestroy(*pStream);
            delete pStream;
        }
    };

    inline std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> makeCudaStream()
    {
        std::unique_ptr<cudaStream_t, decltype(StreamDeleter)> pStream(new cudaStream_t, StreamDeleter);
        if (cudaStreamCreateWithFlags(pStream.get(), cudaStreamNonBlocking) != cudaSuccess)
        {
            pStream.reset(nullptr);
        }

        return pStream;
    }


    TRTEngine::TRTEngine(const EngineParameter &param) : AbstractEngine(param)
    {
        el::Loggers::getLogger(TRT_LOG);
        m_engine = nullptr;
        m_runtime = nullptr;
        m_context = nullptr;

        m_model_name = param.m_model_name;
        m_backend_name = param.m_engine_type;
        m_input_node_name.assign(param.m_input_node_name.begin(), param.m_input_node_name.end());
        m_output_node_name.assign(param.m_output_node_name.begin(), param.m_output_node_name.end());
        m_input_nodes = param.m_input_nodes;
        

        // private
        m_device_id = param.m_device_id;

        cudaSetDevice(m_device_id);


        init(param);
        cudaStreamCreate(&m_stream);
    }





    TRTEngine::~TRTEngine()
    {
        cudaStreamDestroy(m_stream);
        m_context.reset();
        m_engine.reset();
        m_runtime.reset();        
    }

    int TRTEngine::init(const EngineParameter& param)
    {        
        std::string ext = get_file_extension(param.m_model_path) ;
        std::string engine_file;
        if(ext == ".onnx")
        {
            engine_file = param.m_model_path + ".trt";
        }
        else if(ext == ".trt")
        {
            engine_file = param.m_model_path;
        }
        else
        {
            CLOG(ERROR, TRT_LOG) << "model file type not support!";
            return ZJV_STATUS_ERROR;
        }


        

        std::fstream existEngine;
        existEngine.open(engine_file, std::ios::in);
        if (existEngine.is_open()) 
        {
            std::string cached_engine = "";

            
            while (existEngine.peek() != EOF) 
            {
                std::stringstream buffer;
                buffer << existEngine.rdbuf();
                cached_engine.append(buffer.str());
            }
            existEngine.close();

            m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
            CHECK_PTR(m_runtime);

            m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size()), InferDeleter());
            CHECK_PTR(m_engine);

            m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
            CHECK_PTR(m_context);

            std::cout << "deserialize done" << std::endl;

        } 
        else 
        {
            // 构造器
            auto builder = UniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
            CHECK_PTR(builder);

            const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
            auto network = UniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
            CHECK_PTR(network);

            // 解析onnx权重文件
            auto parser = UniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
            CHECK_PTR(parser);

            bool parsed = parser->parseFromFile(param.m_model_path.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
            if (!parsed)
            {
                return false;
            }

            // 设置配置参数
            auto config = UniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
            CHECK_PTR(config);
            config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);
            if (param.m_fp16)
            {
                config->setFlag(nvinfer1::BuilderFlag::kFP16);
            }
            if (param.m_int8)
            {
                config->setFlag(nvinfer1::BuilderFlag::kINT8);
                setAllDynamicRanges(network.get(), 127.0F, 127.0F);
            }

            // enableDLA(builder.get(), config.get(), mParams.dlaCore);

            // CUDA stream used for profiling by the builder.
            auto profileStream = makeCudaStream();
            CHECK_PTR(profileStream.get());
            
            config->setProfileStream(*profileStream);

            UniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
            CHECK_PTR(plan);


            std::ofstream file;

            file.open(engine_file, std::ios::binary | std::ios::out);
            std::cout << "writing engine file..." << std::endl;
            file.write((const char *) plan->data(), plan->size());
            std::cout << "save engine file done" << std::endl;
            file.close();

            m_runtime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
            CHECK_PTR(m_runtime);

            m_engine = std::shared_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()), InferDeleter());
            CHECK_PTR(m_engine);

            m_context = std::shared_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
            CHECK_PTR(m_context);

        }
        

        int no = m_engine->getNbIOTensors();
        for(int i = 0; i < no ; i++)
        {
            std::string io_name = m_engine->getIOTensorName(i);
            nvinfer1::Dims dims = m_engine->getTensorShape(io_name.c_str());
            CLOG(INFO, TRT_LOG) << "IO Tensor Name: " << io_name;
            std::string shape = "" ;
            for(int j = 0; j < dims.nbDims; j++)
            {
                shape = shape + " " + std::to_string(dims.d[j]);
            }
            CLOG(INFO, TRT_LOG) << "IO Tensor Shape: " << shape;
        }

        return ZJV_STATUS_OK;
    }

    int TRTEngine::forward(std::vector<FBlob> &inputs, std::vector<FBlob> &outputs)
    {
        int batchSize = 1;
        for(auto &input: inputs)
        {
            nvinfer1::Dims dims;
            dims.nbDims = input.shape().size();
            batchSize = input.shape()[0];
            for(int i = 0; i < dims.nbDims; i++)
            {
                dims.d[i] = input.shape()[i];
            }
            input.set_device_id(m_device_id);
            m_context->setTensorAddress(input.name_.c_str(), input.mutable_gpu_data());
            m_context->setInputShape(input.name_.c_str(), dims);
            // CLOG(INFO, TRT_LOG) << "input IO Tensor: " << input.name_;
        }

        for(int i = 0; i < m_output_node_name.size(); i++)
        {
            std::string out_name = m_output_node_name[i];
            nvinfer1::Dims dims = m_engine->getTensorShape(out_name.c_str());
            std::vector<int> shape;
            shape.push_back(batchSize);
            for(int j = 1; j < dims.nbDims; j++)
            {                
                shape.push_back(dims.d[j]);                
            }
            FBlob output(shape);
            output.name_ = out_name;
            outputs.push_back(output);
        }

        for(auto &output : outputs)
        {
            output.set_device_id(m_device_id);
            m_context->setTensorAddress(output.name_.c_str(), output.mutable_gpu_data());
            // CLOG(INFO, TRT_LOG) << "output IO Tensor: " << output.name_;
        }

        
        m_context->enqueueV3(m_stream);
        cudaStreamSynchronize(m_stream);
        

        // m_buffers[0] = inputs[0].mutable_gpu_data();

        return ZJV_STATUS_OK;
    }

REGISTER_ENGINE_CLASS(TRT)

} // ZJVIDEO

#endif // Enable_TRT