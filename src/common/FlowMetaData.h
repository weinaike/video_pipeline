

#ifndef ZJVIDEO_FLOWEXTRADATA_H
#define ZJVIDEO_FLOWEXTRADATA_H


#include "BaseData.h"
#include <vector>
#include <memory>


namespace ZJVIDEO {

class FrameData;
// pipeline中流转的数据流
class FlowData : public BaseData 
{
public:
    explicit FlowData(): BaseData(ZJV_DATATYPE_FLOW) {}
    ~FlowData() override = default;


    // 流转的帧不能被修改
    std::shared_ptr<const FrameData > frame; //帧数据
    // 可以添加修改随帧数据
    std::vector<std::shared_ptr<const BaseData> >  m_extras; // 额外数据
};


// 检测结果，分类结果等
class ExtraData : public BaseData 
{
public:
    explicit ExtraData(): BaseData(ZJV_DATATYPE_EXTRA) {}
    ~ExtraData() override = default;

};


}  // namespace ZJVIDEO

#endif
