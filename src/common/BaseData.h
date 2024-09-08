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

#include "public/PublicData.h"
namespace ZJVIDEO {

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
