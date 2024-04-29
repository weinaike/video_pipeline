#ifndef ZJVIDEO_INFERNODE_H
#define ZJVIDEO_INFERNODE_H

#include "../BaseNode.h"
#include "backend/EngineFactory.h"
#include "InferDefine.h"
#include "PreProcessor.h"
#include "PostProcessor.h"

namespace ZJVIDEO {

#define INFER_LOG "InferNode"
class InferNode : public BaseNode {

public:

    InferNode(const NodeParam & param);
    virtual ~InferNode();
    InferNode() = delete;

protected:
    //parse,解析配置文件
    virtual int parse_configure(std::string cfg_file); 
    //根据配置文件， 初始化对象,输入输出队列
    virtual int init();         
            
    virtual int process_batch( const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                                std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch);

    // the 1st step, MUST implement in specific derived class.
    // prepare data for infer, fetch frames from frame meta.
    virtual int prepare(const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                            std::vector<std::shared_ptr<FrameROI>>  &frame_rois);
    
    // the 2nd step, has a default implementation.
    // preprocess data, such as normalization, mean substract. 
    // load to engine's inputs
    virtual int preprocess(std::vector<std::shared_ptr<FrameROI>>  &frame_rois, std::vector<FBlob> &inputs); 

    // the 3rd step, has a default implementation.
    // infer and retrive raw outputs.
    virtual int infer(std::vector<FBlob> &inputs, std::vector<FBlob> &outputs); 
    
    // the 4th step, MUST implement in specific derived class.
    // postprocess on raw outputs and create/update something back to frame meta again.
    virtual int postprocess(std::vector<FBlob> & outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois);

    virtual int summary(const std::vector<std::shared_ptr<FrameROI>>  &frame_rois, 
                std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch);


protected:
    std::shared_ptr<AbstractEngine>                 m_engine;
    EngineParameter                                 m_engine_param;
    std::vector<std::shared_ptr<PreProcessor>>      m_img_preprocs; 
    std::vector<PreProcessParameter>                m_img_preproc_params; 
    std::vector<std::shared_ptr<PostProcessor>>     m_postprocess;
    int                                             m_device_id;


}; // class InferNode

} // namespace ZJVIDEO


#endif