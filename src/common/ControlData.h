//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_CONTROLDATA_H
#define VIDEOPIPELINE_CONTROLDATA_H

#include "BaseData.h"
namespace ZJVIDEO {

    enum ControlType {
        ZJV_CONTROLTYPE_UNKNOWN = 0,
        ZJV_CONTROLTYPE_VIDEO_RECORD,  // 视频录制
        ZJV_CONTROLTYPE_IMAGE_RECORD,  // 图片录制
        ZJV_CONTROLTYPE_MAX
    };

    // 用于存储控制数据的类
    class ControlData : public BaseData {
    public:
        explicit ControlData(ControlType control_type)
            : BaseData(ZJV_DATATYPE_CONTROL), m_control_type(control_type) {}

        ~ControlData() override = default;

    private:
        ControlType m_control_type;  // 控制类型
    public:
        ControlType get_control_type() const {
            return m_control_type;
        }
    };

}  // namespace Data
#endif  // VIDEOPIPELINE_CONTROLDATA_H
