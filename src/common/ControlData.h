//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_CONTROLDATA_H
#define VIDEOPIPELINE_CONTROLDATA_H

#include "BaseData.h"
namespace ZJVIDEO {

enum ControlType {
    ZJV_CONTROLTYPE_UNKNOWN = 0,
    ZJV_CONTROLTYPE_VIDEO_RECORD,       // 视频录制
    ZJV_CONTROLTYPE_IMAGE_RECORD,       // 图片录制
    ZJV_CONTROLTYPE_GET_FPS,            // 获取帧率
    ZJV_CONTROLTYPE_SET_FPS,            // 设置帧率
    ZJV_CONTROLTYPE_SET_LOGGER_LEVEL,   // 设置日志级别
    ZJV_CONTROLTYPE_SET_RUN_MODE,       // 设置运行模式
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

class GetFPSControlData : public ControlData {
public:
    explicit GetFPSControlData(ControlType control_type = ZJV_CONTROLTYPE_GET_FPS)
        : ControlData(control_type){}

    ~GetFPSControlData() override = default;

private:
    float m_fps;
public:
    float get_fps() const {
        return m_fps;
    }

    int set_fps(float fps)  {
        m_fps = fps;
        return 0;
    }
};

enum LoggerLevel {
    ZJV_LOGGER_LEVEL_FATAL = 0,  // fatal
    ZJV_LOGGER_LEVEL_ERROR,  // error
    ZJV_LOGGER_LEVEL_WARN,  // warn
    ZJV_LOGGER_LEVEL_INFO,  // info  
    ZJV_LOGGER_LEVEL_DEBUG  // debug
};

class SetLoggerLevelControlData : public ControlData {

public:
    explicit SetLoggerLevelControlData(ControlType control_type = ZJV_CONTROLTYPE_SET_LOGGER_LEVEL)
        : ControlData(control_type){}

    ~SetLoggerLevelControlData() override = default;

private:
    int m_level;
public:
    int get_level() const {
        return m_level;
    }

    int set_level(int level)  {
        m_level = level;
        return 0;
    }
};


enum PipelineRunMode {
    ZJV_PIPELINE_RUN_MODE_LIVING,  // 实时在线
    ZJV_PIPELINE_RUN_MODE_RECORDED,  // 预先录制
};

class SetRunModeControlData : public ControlData {

public:
    explicit SetRunModeControlData(ControlType control_type = ZJV_CONTROLTYPE_SET_RUN_MODE)
        : ControlData(control_type){}

    ~SetRunModeControlData() override = default;

private:
    int m_mode;
public:
    int get_mode() const {
        return m_mode;
    }

    int set_mode(int mode)  {
        m_mode = mode;
        return 0;
    }
};


}  // namespace Data
#endif  // VIDEOPIPELINE_CONTROLDATA_H
