//
// Created by lijin on 2023/8/3.
//

#ifndef ZJVIDEO_BASEDATA_H
#define ZJVIDEO_BASEDATA_H

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <map>
#include <vector>

namespace ZJVIDEO {

enum BaseDataType {
    ZJV_DATATYPE_UNKNOWN = 0,
    ZJV_DATATYPE_VIDEO,    // 视频数据
    ZJV_DATATYPE_FRAME,    // 帧数据
    ZJV_DATATYPE_CONTROL,  // 控制数据
    ZJV_DATATYPE_CONFIG,   // 配置数据
    ZJV_DATATYPE_EXTRA,    // 额外数据，检测结果，识别结果等
    ZJV_DATATYPE_FLOW,     // 流数据
    ZJV_DATATYPE_DETECTRESULT,    //检测结果，实例分割结果
    ZJV_DATATYPE_DETECTRESULT_TRACK,    //跟踪结果
    ZJV_DATATYPE_CLASSIFYRESULT,  //分类结果
    ZJV_DATATYPE_SEGMENTRESULT,   //语义分割结果
    ZJV_DATATYPE_IMAGECACHE,   //图像缓存
    ZJV_DATATYPE_FEATURECACHE, //特征缓存



    ZJV_DATATYPE_EVENT = 1000,    // 事件数据
    ZJV_DATATYPE_EVENT_WELDING,    // 焊接事件数据

    ZJV_DATATYPE_MAX
};

// 用于存储数据的基类
class BaseData {
public:
    BaseData() = delete;

    explicit BaseData(BaseDataType data_type) : data_type(data_type) 
    {
        create_time     = std::chrono::system_clock::now();
    }

    virtual ~BaseData() = default;
    virtual int append(std::shared_ptr<BaseData>& data) { return 0; }

    BaseDataType get_data_type() const 
    {
        return data_type;
    }

    BaseDataType                          data_type;    // 数据类型
    std::chrono::system_clock::time_point create_time;  // 数据创建时间
    std::string                           data_name;    // 数据名称/数据来

};


class DataRegister
{
public:

    typedef std::shared_ptr<BaseData> (*Creator)();
    typedef std::map<std::string, Creator> CreatorRegistry;

    static CreatorRegistry& Registry() 
    {
        static CreatorRegistry* g_registry_data_ = new CreatorRegistry();
        return *g_registry_data_;
    }

    // Adds a creator.
    static void AddCreator(const std::string& type, Creator creator) 
    {
        CreatorRegistry& registry = Registry();
        // 判断是否已经存在
        if (registry.find(type) != registry.end()) 
        {
            std::cout << "Data type " << type << " already registered." << std::endl;
            return;
        }
        registry[type] = creator;
    }

    // Get a BaseData using a type.
    static std::shared_ptr< BaseData > CreateData(const std::string& type) 
    {
        CreatorRegistry& registry = Registry();
        // 判断是否存在
        if (registry.find(type) == registry.end()) 
        {
            std::cout << "Unknown BaseData type: " << type << " \nall: " << DataTypeListString()<< std::endl;
            return nullptr;
        }
        return registry[type]();
    }

    static std::vector<std::string> DataTypeList() 
    {
        CreatorRegistry& registry = Registry();
        std::vector<std::string> data_types;
        // 遍历map,提取key，放入data_types

        for (typename CreatorRegistry::iterator iter = registry.begin();
                    iter != registry.end(); ++iter) 
        {
            data_types.push_back(iter->first);
        }
        return data_types;
    }

 private:
    // Node registry should never be instantiated - everything is done with its
    // static variables.
    DataRegister() {}

    static std::string DataTypeListString() 
    {
        std::vector<std::string> data_types = DataTypeList();
        std::string data_types_str;
        for (std::vector<std::string>::iterator iter = data_types.begin();
            iter != data_types.end(); ++iter) 
        {
            if (iter != data_types.begin()) {
                data_types_str += ", ";
            }
            data_types_str += *iter;
        }
        return data_types_str;
    }
};



class DataRegisterer 
{
public:
    DataRegisterer(const std::string& type, 
        std::shared_ptr<BaseData> (*creator)()) 
    {
        std::cout << "Registering data type: " << type << std::endl;
        DataRegister::AddCreator(type, creator);
    }

};



#define REGISTER_DATA_CREATOR(type, creator)                                    \
    static DataRegisterer g_creator_data_f_##type(#type, creator);              \


#define REGISTER_DATA_CLASS(type)                                               \
    std::shared_ptr<BaseData> Creator_##type##Data()                            \
    {                                                                           \
        return std::shared_ptr<BaseData>(new type##Data());                     \
    }                                                                           \
    REGISTER_DATA_CREATOR(type, Creator_##type##Data)


}  // namespace ZJVIDEO

#endif  // ZJVIDEO_BASEDATA_H
