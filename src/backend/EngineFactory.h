



#ifndef ZJVIDEO_ENGINEFACTORY_H
#define ZJVIDEO_ENGINEFACTORY_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <iostream>

#include "common/StatusCode.h"
#include "AbstractEngine.h"

namespace ZJVIDEO {


class EngineRegister
{
public:

    typedef std::shared_ptr<AbstractEngine> (*Creator)(const EngineParameter &);
    typedef std::map<std::string, Creator> CreatorRegistry;

    static CreatorRegistry& Registry() 
    {
        static CreatorRegistry* g_registry_engine_ = new CreatorRegistry();
        return *g_registry_engine_;
    }

    // Adds a creator.
    static void AddCreator(const std::string& type, Creator creator) 
    {
        CreatorRegistry& registry = Registry();
        // 判断是否已经存在
        if (registry.find(type) != registry.end()) 
        {
            std::cout << "Engine type " << type << " already registered." << std::endl;
            return;
        }
        registry[type] = creator;
    }

    // Get a  infer engine using a EngineParameter.
    static std::shared_ptr< AbstractEngine > CreateEngine(const EngineParameter& param) 
    {
        std::cout<<"Creating Engine  " << param.m_engine_type << std::endl;

        const std::string& type = param.m_engine_type;
        CreatorRegistry& registry = Registry();
        // 判断是否存在
        if (registry.find(type) == registry.end()) 
        {
            std::cout << "Unknown engine type: " << type << " \nall Engine: [" << EngineTypeListString() << "]"<< std::endl;
            return nullptr;
        }
        return registry[type](param);
    }

    static std::vector<std::string> EngineTypeList() 
    {
        CreatorRegistry& registry = Registry();
        std::vector<std::string> engine_types;
        // 遍历map,提取key，放入node_types

        for (typename CreatorRegistry::iterator iter = registry.begin();
                    iter != registry.end(); ++iter) 
        {
            engine_types.push_back(iter->first);
        }
        return engine_types;
    }

 private:
    // Node registry should never be instantiated - everything is done with its
    // static variables.
    EngineRegister() {}

    static std::string EngineTypeListString() 
    {
        std::vector<std::string> engine_types = EngineTypeList();
        std::string engine_types_str;
        for (std::vector<std::string>::iterator iter = engine_types.begin();
            iter != engine_types.end(); ++iter) 
        {
            if (iter != engine_types.begin()) {
                engine_types_str += ", ";
            }
            engine_types_str += *iter;
        }
        return engine_types_str;
    }
};



class EngineRegisterer 
{
public:
    EngineRegisterer(const std::string& type, 
        std::shared_ptr<AbstractEngine > (*creator)(const EngineParameter&)) 
    {
        std::cout << "Registering engine type: " << type << std::endl;
        EngineRegister::AddCreator(type, creator);
    }

};



#define REGISTER_ENGINE_CREATOR(type, creator)                                              \
    static EngineRegisterer g_creator_engine_##type(#type, creator);                        \

#define REGISTER_ENGINE_CLASS(type)                                                         \
    std::shared_ptr<AbstractEngine> Creator_##type##Engine(const EngineParameter& param)    \
    {                                                                                       \
        return std::shared_ptr<AbstractEngine>(new type##Engine(param));                    \
    }                                                                                       \
    REGISTER_ENGINE_CREATOR(type, Creator_##type##Engine)

}  // namespace ZJVIDEO

#endif  // ZJVIDEO_ENGINEFACTORY_H