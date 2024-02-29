//
// Created by lijin on 2023/12/21.
//

#ifndef VIDEOPIPELINE_CONFIGDATA_H
#define VIDEOPIPELINE_CONFIGDATA_H

#include "BaseData.h"

namespace ZJVIDEO {

    enum ConfigType {
        ZJV_CONFIGTYPE_UNKNOWN = 0,
        ZJV_CONFIGTYPE_ROI_CONFIG,       // ROI配置
        ZJV_CONFIGTYPE_ALARM_CONFIG,     // 报警配置
        ZJV_CONFIGTYPE_ANALYSIS_CONFIG,  // 分析配置
        ZJV_CONFIGTYPE_MAX
    };

    // 用于存储配置数据的类
    class ConfigData : public BaseData {
    public:
        explicit ConfigData(ConfigType config_type) : BaseData(ZJV_DATATYPE_CONFIG), m_config_type(config_type) {}

        ~ConfigData() override = default;

    private:
        ConfigType m_config_type;  // 配置类型
    public:
        ConfigType get_config_type() const {
            return m_config_type;
        }
    };

}  // namespace Data
#endif  // VIDEOPIPELINE_CONFIGDATA_H
