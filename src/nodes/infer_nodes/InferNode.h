#ifndef ZJVIDEO_INFERNODE_H
#define ZJVIDEO_INFERNODE_H

#include "../BaseNode.h"
#include "../backend/EngineFactory.h"
#include "../common/Blob.h"

namespace ZJVIDEO {

struct FrameROI
{
    int input_vector_id;
    std::shared_ptr<const FrameData> frame;
    // 原图坐标系下的roi
    Rect roi;                              
    // 网络输入宽
    int input_width;                        
    // 网络输入高
    int input_height;                       
    // 缩放比例x，roi宽/网络输入宽
    float scale_x;
    // 缩放比例y, roi高/网络输入高        
    float scale_y;
    // 对于letterbox的缩放模式，填充起始点x，y          
    int padx;
    int pady;
    // 模型推理结果，可以支持多种结果同时输出
    std::vector< std::shared_ptr<BaseData>> result;  

};



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
    virtual int postprocess(const std::vector<FBlob> & outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois);

    virtual int summary(std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch);


protected:
    std::shared_ptr<AbstractEngine>     m_engine;
    EngineParameter                     m_engine_param;

}; // class InferNode

} // namespace ZJVIDEO


#endif