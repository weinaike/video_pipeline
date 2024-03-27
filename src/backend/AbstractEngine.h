
#ifndef ZJVIDEO_ABSTRACTENGINE_H
#define ZJVIDEO_ABSTRACTENGINE_H

#include <string>
#include <memory>
#include <vector>
#include "common/NodeParam.h"
#include "common/ThreadSaveQueue.h"
#include "common/StatusCode.h"
#include <map>
#include "common/StatusCode.h"
#include "common/Blob.h"


namespace ZJVIDEO {


enum EngineID {
    IDLE,
    ONNX = 1,
    MNN = 2,
    NCNN = 3,
    TNN = 4,
    OPENVINO = 5,
    PADDLE_LITE = 6,
    TRT = 7

};  /*!< Engine类型*/

enum Device { ZJV_DEVICE_CPU = 0, ZJV_DEVICE_GPU = 1}; /*!< 模型加载device*/

struct EngineParameter {
    std::string                                 m_engine_type;          /*!< 引擎名称*/
    std::string                                 m_model_name;           /*!< 模型名称*/
    int                                         m_max_batch_size = 8;   /*!< 最大batch size*/
    bool                                        m_encrypt   = false;    /*!< 模型是否加密*/
    int                                         m_device_id = -1;
    bool                                        m_dynamic   = false;
    bool                                        m_int8      = false;                  //!< Allow runnning the network in Int8 mode.
    bool                                        m_fp16      = false;                  //!< Allow running the network in FP16 mode.
    std::string                                 m_model_path;
    std::string                                 m_param_path;
    std::string                                 m_version;
    // To Do
    // 有些模型可能有多个输入输出节点，后续再修改
    std::map<std::string, std::vector<int>>     m_input_nodes;          /*!< 输入节点信息 {name: shape}*/
    std::vector<std::string>                    m_input_node_name;      /*!< 输入节点名称*/
    std::vector<std::string>                    m_output_node_name;     /*!< 输出节点名称*/
    std::vector<std::string>                    m_register_layers       /*!< 自定义layer*/;

};


class AbstractEngine {

public:
    // 1. 通过工厂方法创建实例
    AbstractEngine(const EngineParameter & param){};
    AbstractEngine() = delete;
    virtual ~AbstractEngine() = default;

    virtual int init(const EngineParameter&) = 0;
    // virtual int forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) = 0;
    virtual int forward(std::vector<FBlob> &inputs, std::vector<FBlob> &outputs) = 0;
    std::vector<std::string> m_output_node_name;
    std::vector<std::string> m_input_node_name;
    std::map<std::string, std::vector<int>> m_input_nodes;  /*!< 输入节点信息*/
    bool m_dynamic=false;
    std::string m_model_name = "default";
    std::string m_backend_name = "default";

  
}; // class AbstractEngine

}  // namespace ZJVIDEO

#endif  // ZJVIDEO_ABSTRACTENGINE_H