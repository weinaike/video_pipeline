#ifndef _ZJV_TENSORRT_ENGINE_H_
#define _ZJV_TENSORRT_ENGINE_H_
#define Enable_TRT
#ifdef Enable_TRT

#include <string>
#include <vector>
#include "EngineFactory.h"
#include "TensorRT/NvInfer.h"
// #include "TensorRT/NvInferPlugin.h"
#include "TensorRT/NvInferRuntime.h"
#include <cuda_runtime_api.h>

#include "assert.h"
#include <unordered_map>



namespace ZJVIDEO{


class Logger : public nvinfer1::ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class Logger logger;

class TRTEngine: public AbstractEngine{

public:
    TRTEngine(const EngineParameter & param);
    ~TRTEngine() override;
    virtual int init(const EngineParameter&) override;    
    virtual int forward(std::vector<FBlob> &inputs, std::vector<FBlob> &outputs) override;

private:

    std::shared_ptr<nvinfer1::IRuntime> m_runtime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine; //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::IExecutionContext> m_context; //!< The TensorRT engine used to run the network
    cudaStream_t m_stream;
    int                                  m_device_id = 0;
};



// Ensures that every tensor used by a network has a dynamic range set.
//
// All tensors in a network must have a dynamic range specified if a calibrator is not used.
// This function is just a utility to globally fill in missing scales and zero-points for the entire network.
//
// If a tensor does not have a dyanamic range set, it is assigned inRange or outRange as follows:
//
// * If the tensor is the input to a layer or output of a pooling node, its dynamic range is derived from inRange.
// * Otherwise its dynamic range is derived from outRange.
//
// The default parameter values are intended to demonstrate, for final layers in the network,
// cases where dynamic ranges are asymmetric.
//
// The default parameter values choosen arbitrarily. Range values should be choosen such that
// we avoid underflow or overflow. Also range value should be non zero to avoid uniform zero scale tensor.
inline void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange = 2.0F, float outRange = 4.0F)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                assert(input->setDynamicRange(-inRange, inRange));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    assert(output->setDynamicRange(-inRange, inRange));
                }
                else
                {
                    assert(output->setDynamicRange(-outRange, outRange));
                }
            }
        }
    }
}

inline void setDummyInt8DynamicRanges(const nvinfer1::IBuilderConfig* c, nvinfer1::INetworkDefinition* n)
{
    // Set dummy per-tensor dynamic range if Int8 mode is requested.
    if (c->getFlag(nvinfer1::BuilderFlag::kINT8))
    {
        std::cout << "Int8 calibrator not provided. Generating dummy per-tensor dynamic range. Int8 accuracy is not guaranteed." << std::endl;
        setAllDynamicRanges(n);
    }
}

inline void enableDLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allowGPUFallback = true)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
    }
}


} //ZJVIDEO

#endif //Enable_TRT    

#endif  // _ZJV_TENSORRT_ENGINE_H_
