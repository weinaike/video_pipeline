

#ifndef ZJVIDEO_FLOWEXTRADATA_H
#define ZJVIDEO_FLOWEXTRADATA_H

#include "BaseData.h"
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include "ExtraData.h"

#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>
namespace ZJVIDEO
{
    class FrameData;
    class ExtraData;
    // pipeline中流转的数据流
    class FlowData : public BaseData
    {
    public:
        explicit FlowData(const std::shared_ptr<FrameData> &data) : BaseData(ZJV_DATATYPE_FLOW)
        {
            frame = data;
            video = nullptr;
            data_name = "Flow";
            m_flow_id = generateRandomID();
        }

        explicit FlowData(const std::shared_ptr<VideoData> &data) : BaseData(ZJV_DATATYPE_FLOW)
        {
            video = data;
            frame = nullptr;
            data_name = "Flow";
            m_flow_id = generateRandomID();
        }
        ~FlowData() override = default;

        // 添加额外数据
        inline int push_back(std::vector<std::pair<std::string, std::shared_ptr<const BaseData>>> &datas)
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            for (const auto &data : datas)
            {
                // 如果tag已经存在，就不添加
                auto it = m_extras.find(data.first);
                if (it != m_extras.end())
                {
                    std::cout << data.first << " is existed in m_extras" << std::endl;
                    return ZJV_STATUS_ERROR;
                }
                else
                {
                    m_extras.emplace(data);
                }
            }

            return ZJV_STATUS_OK;
        }
        // 获取额外数据
        inline int get_extras(const std::vector<std::string> &tags, std::vector<std::shared_ptr<const BaseData>> &data)
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            for (const auto &tag : tags)
            {
                auto it = m_extras.find(tag);
                if (it == m_extras.end())
                {
                    std::cout << tag << " is not existed in m_extras" << std::endl;
                    return ZJV_STATUS_ERROR;
                }
                data.push_back(std::ref(it->second)); // Pass the vector by reference
            }
            return ZJV_STATUS_OK;
        }
        inline int get_all_extras(std::vector<std::shared_ptr<const BaseData>> &data)
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            for (const auto &m_extra : m_extras)
            {
                auto it = m_extra.second;
                data.push_back(std::ref(it)); // Pass the vector by reference
            }
            return ZJV_STATUS_OK;
        }

        inline int get_node_extras(std::string node_name, std::vector<std::shared_ptr<const BaseData>> &data)
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            for (const auto &m_extra : m_extras)
            {
                if (m_extra.first.compare(0, node_name.size(), node_name) != 0) continue;
 
                auto it = m_extra.second;
                data.push_back(std::ref(it)); // Pass the vector by reference
            }
            return ZJV_STATUS_OK;
        }
        // 判断额外数据是否存在
        inline bool has_extras(const std::vector<std::string> &tags)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            return std::all_of(tags.begin(), tags.end(), [this](const std::string &tag)
                               { return m_extras.count(tag) > 0; });
        }
        // 仅在送入pipeline时设置一次
        inline int set_channel_id(int channel_id)
        {
            m_channel_id = channel_id;
            return ZJV_STATUS_OK;
        }
        inline int get_channel_id()
        {
            return m_channel_id;
        }
        inline std::shared_ptr<const FrameData> get_frame()
        {
            return frame;
        }
        inline std::shared_ptr<const VideoData> get_video()
        {
            return video;
        }
        inline void debug()
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            std::cout << "------------" << m_flow_id <<"-----------" <<std::endl;
            for (const auto &extra : m_extras)
            {
                std::cout << extra.first << std::endl;
            }
            std::cout << "------------flow data-----------" <<std::endl;
        }

        inline std::string get_flow_id()
        {
            return m_flow_id;
        }

        // 生成随机ID
        inline std::string generateRandomID() 
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, 15);
            std::stringstream ss;

            // 生成32位的16进制数字符串
            for (int i = 0; i < 32; i++) {
                ss << std::hex << dis(gen);
            }

            return ss.str();
        }

    private:
        // 流转过程中，create_time和指针地址是标识
        std::shared_ptr<const FrameData> frame; // 帧数据
        std::shared_ptr<const VideoData> video; // 帧数据
        // 可以添加修改随帧数据
        std::map<std::string, std::shared_ptr<const BaseData>> m_extras; // 额外数据
        // 锁
        std::mutex m_mutex;
        int m_channel_id = -1;
        std::string m_flow_id; 
    };

} // namespace ZJVIDEO

#endif
