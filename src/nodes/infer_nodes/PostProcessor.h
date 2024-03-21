#ifndef __ZJV_POSTPROCESS_H__
#define __ZJV_POSTPROCESS_H__
#include "InferDefine.h"

namespace ZJVIDEO
{
    class PostProcessor
    {
    public:
        PostProcessor() = default;
        ~PostProcessor() = default;

        virtual int parse_json(const nlohmann::json & j) = 0; 
        virtual int run(std::vector<FBlob> &outputs, std::vector<std::shared_ptr<FrameROI>> &frame_rois) = 0;
    protected:
        std::string m_output_data_type;
        std::string m_post_type;
        std::vector<std::string> m_input_names;
        std::vector<int> m_main_categories;
        std::vector<int> m_sub_categories;
    };

    class PostRegister
    {
    public:
        typedef std::shared_ptr<PostProcessor> (*Creator)();
        typedef std::map<std::string, Creator> CreatorRegistry;

        static CreatorRegistry &Registry()
        {
            static CreatorRegistry *g_registry_post_ = new CreatorRegistry();
            return *g_registry_post_;
        }

        // Adds a creator.
        static void AddCreator(const std::string &type, Creator creator)
        {
            CreatorRegistry &registry = Registry();
            // 判断是否已经存在
            if (registry.find(type) != registry.end())
            {
                std::cout << "post process type " << type << " already registered." << std::endl;
                return;
            }
            registry[type] = creator;
        }

        // Get a PostProcessor using a type.
        static std::shared_ptr<PostProcessor> CreatePost(const std::string &type)
        {
            CreatorRegistry &registry = Registry();
            // 判断是否存在
            if (registry.find(type) == registry.end())
            {
                std::cout << "Unknown PostProcessor type: " << type << " \nall: " << PostTypeListString() << std::endl;
                return nullptr;
            }
            return registry[type]();
        }

        static std::vector<std::string> PostTypeList()
        {
            CreatorRegistry &registry = Registry();
            std::vector<std::string> post_types;
            // 遍历map,提取key，放入data_types

            for (typename CreatorRegistry::iterator iter = registry.begin();
                 iter != registry.end(); ++iter)
            {
                post_types.push_back(iter->first);
            }
            return post_types;
        }

    private:
        // Node registry should never be instantiated - everything is done with its
        // static variables.
        PostRegister() {}

        static std::string PostTypeListString()
        {
            std::vector<std::string> post_types = PostTypeList();
            std::string post_types_str;
            for (std::vector<std::string>::iterator iter = post_types.begin();
                 iter != post_types.end(); ++iter)
            {
                if (iter != post_types.begin())
                {
                    post_types_str += ", ";
                }
                post_types_str += *iter;
            }
            return post_types_str;
        }
    };

    class PostRegisterer
    {
    public:
        PostRegisterer(const std::string &type,
                       std::shared_ptr<PostProcessor> (*creator)())
        {
            std::cout << "Registering post process type: " << type << std::endl;
            PostRegister::AddCreator(type, creator);
        }
    };

#define REGISTER_POST_CREATOR(type, creator) \
    static PostRegisterer g_creator_post_f_##type(#type, creator);

#define REGISTER_POST_CLASS(type)                                         \
    std::shared_ptr<PostProcessor> Creator_##type##PostProcessor()        \
    {                                                                     \
        return std::shared_ptr<PostProcessor>(new type##PostProcessor()); \
    }                                                                     \
    REGISTER_POST_CREATOR(type, Creator_##type##PostProcessor)

} // namespace ZJVideo

#endif // __ZJV_POSTPROCESS_H__