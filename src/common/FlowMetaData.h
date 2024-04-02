

#ifndef ZJVIDEO_FLOWEXTRADATA_H
#define ZJVIDEO_FLOWEXTRADATA_H

#include "BaseData.h"
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include "ExtraData.h"

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
        }

        explicit FlowData(const std::shared_ptr<VideoData> &data) : BaseData(ZJV_DATATYPE_FLOW)
        {
            video = data;
            frame = nullptr;
            data_name = "Flow";
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
            std::cout << "------------flow data-----------" << m_extras.size() << std::endl;
            for (const auto &extra : m_extras)
            {
                std::cout << "------------flow data-----------" << extra.first << std::endl;
            }
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
    };

} // namespace ZJVIDEO

#endif
