

#ifndef ZJVIDEO_FLOWEXTRADATA_H
#define ZJVIDEO_FLOWEXTRADATA_H


#include "BaseData.h"
#include <vector>
#include <memory>
#include <map>
#include <algorithm>

namespace ZJVIDEO {

class FrameData;
class ExtraData;
// pipeline中流转的数据流
class FlowData : public BaseData 
{
public:
    explicit FlowData(const std::shared_ptr<FrameData> &data ): BaseData(ZJV_DATATYPE_FLOW) 
    {
        frame = data;
    }
    ~FlowData() override = default;

    // 流转的帧不能被修改， 流转过程中通过FrameData的camera_id和Frame_id来标识唯一的帧
    std::shared_ptr<const FrameData > frame; //帧数据
     // 添加额外数据
    inline int push_back(std::vector<std::pair<std::string, std::shared_ptr<const ExtraData> > > &datas )
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        for (const auto & data : datas)
        {
            // 如果tag已经存在，就不添加
            auto it = m_extras.find(data.first);
            if (it != m_extras.end()) {
                std::cout<< data.first << " is existed in m_extras" <<std::endl; 
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
    inline int get_extras(const std::vector< std::string > & tags, std::vector<std::shared_ptr<const ExtraData>>& data)
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        for(const auto & tag :tags)
        {
            auto it = m_extras.find(tag);
            if (it == m_extras.end()) {
                std::cout << tag << " is not existed in m_extras" << std::endl; 
                return ZJV_STATUS_ERROR;
            } 
            data.push_back(std::ref(it->second)); // Pass the vector by reference
        }
        return ZJV_STATUS_OK;
    }
    // 判断额外数据是否存在
    inline bool has_extras(const std::vector<std::string> & tags)
    {   
        std::lock_guard<std::mutex> lock(m_mutex);
        return std::all_of(tags.begin(), tags.end(), [this](const std::string& tag) {return m_extras.count(tag) > 0;});
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

private:
    // 可以添加修改随帧数据
    std::map<std::string, std::shared_ptr<const ExtraData> >  m_extras; // 额外数据
    // 锁
    std::mutex                  m_mutex;
    int                         m_channel_id = -1 ;
};


// 检测结果，分类结果等
class ExtraData : public BaseData 
{
public:
    explicit ExtraData(): BaseData(ZJV_DATATYPE_EXTRA) {}
    ~ExtraData() override = default;

};


}  // namespace ZJVIDEO

#endif
