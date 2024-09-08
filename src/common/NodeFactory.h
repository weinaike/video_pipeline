



#ifndef ZJVIDEO_NODEFACTORY_H
#define ZJVIDEO_NODEFACTORY_H

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <iostream>

#include "ThreadSaveQueue.h"
#include "StatusCode.h"
#include "AbstractNode.h"

namespace ZJVIDEO {


class NodeRegister
{
public:

    typedef std::shared_ptr<AbstractNode> (*Creator)(const NodeParam &);
    typedef std::map<std::string, Creator> CreatorRegistry;

    static CreatorRegistry& Registry() 
    {
        static CreatorRegistry* g_registry_ = new CreatorRegistry();
        return *g_registry_;
    }

    // Adds a creator.
    static void AddCreator(const std::string& type, Creator creator) 
    {
        CreatorRegistry& registry = Registry();
        // 判断是否已经存在
        if (registry.find(type) != registry.end()) 
        {
            std::cout << "Node type " << type << " already registered." << std::endl;
            return;
        }
        registry[type] = creator;
    }

    // Get a layer using a NodeParam.
    static std::shared_ptr< AbstractNode > CreateNode(const NodeParam& param) 
    {
        std::cout<<"Creating Node " << param.m_node_name << std::endl;

        const std::string& type = param.m_node_type;
        CreatorRegistry& registry = Registry();
        // 判断是否存在
        if (registry.find(type) == registry.end()) 
        {
            std::cout << "Unknown node type: " << type << " \nall supported node: [" << NodeTypeListString() <<"]"<< std::endl;
            return nullptr;
        }
        return registry[type](param);
    }

    static std::vector<std::string> NodeTypeList() 
    {
        CreatorRegistry& registry = Registry();
        std::vector<std::string> node_types;
        // 遍历map,提取key，放入node_types

        for (typename CreatorRegistry::iterator iter = registry.begin();
                    iter != registry.end(); ++iter) 
        {
            node_types.push_back(iter->first);
        }
        return node_types;
    }

 private:
    // Node registry should never be instantiated - everything is done with its
    // static variables.
    NodeRegister() {}

    static std::string NodeTypeListString() 
    {
        std::vector<std::string> node_types = NodeTypeList();
        std::string node_types_str;
        for (std::vector<std::string>::iterator iter = node_types.begin();
            iter != node_types.end(); ++iter) 
        {
            if (iter != node_types.begin()) {
                node_types_str += ", ";
            }
            node_types_str += *iter;
        }
        return node_types_str;
    }
};



class NodeRegisterer 
{
public:
    NodeRegisterer(const std::string& type, 
        std::shared_ptr<AbstractNode > (*creator)(const NodeParam&)) 
    {
        std::cout << "Registering node type: " << type << std::endl;
        NodeRegister::AddCreator(type, creator);
    }

};



#define REGISTER_NODE_CREATOR(type, creator)                                    \
    static NodeRegisterer g_creator_f_##type(#type, creator);                   \

#define REGISTER_NODE_CLASS(type)                                               \
    std::shared_ptr<AbstractNode> Creator_##type##Node(const NodeParam& param)  \
    {                                                                           \
        return std::shared_ptr<AbstractNode>(new type##Node(param));            \
    }                                                                           \
    REGISTER_NODE_CREATOR(type, Creator_##type##Node)

}  // namespace ZJVIDEO

#endif  // ZJVIDEO_NODEFACTORY_H