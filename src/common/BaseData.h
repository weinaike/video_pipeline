//
// Created by lijin on 2023/8/3.
//

#ifndef ZJVIDEO_BASEDATA_H
#define ZJVIDEO_BASEDATA_H

#include <chrono>

namespace ZJVIDEO {

    enum BaseDataType {
        ZJV_DATATYPE_UNKNOWN = 0,
        ZJV_DATATYPE_FRAME,    // 帧数据
        ZJV_DATATYPE_CONTROL,  // 控制数据
        ZJV_DATATYPE_CONFIG,   // 配置数据
        ZJV_DATATYPE_MAX
    };

    // 用于存储数据的基类
    class BaseData {
    public:
        BaseData() = delete;

        explicit BaseData(BaseDataType data_type) {
            this->data_type = data_type;
            create_time     = std::chrono::system_clock::now();
        }

        virtual ~BaseData() = default;

        BaseDataType get_data_type() const {
            return data_type;
        }

        BaseDataType                          data_type;    // 数据类型
        std::chrono::system_clock::time_point create_time;  // 数据创建时间
        std::string                           data_name;    // 数据名称/数据来

    };

}  // namespace 

#endif  // ZJVIDEO_BASEDATA_H
