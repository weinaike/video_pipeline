//
// Created by lijin on 2023/12/18.
//

#ifndef ZJVIDEO_THREADSAVEQUEUE_H
#define ZJVIDEO_THREADSAVEQUEUE_H

#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include "BaseData.h"

namespace ZJVIDEO {
// 缓冲队列满时的策略
enum BufferOverStrategy {
    ZJV_QUEUE_DROP_EARLY,  // 丢弃最早的帧
    ZJV_QUEUE_DROP_LATE,   // 丢弃最新的帧
    ZJV_QUEUE_CLEAR,       // 清空缓冲队列
    ZJV_QUEUE_BLOCK        // 堵塞，直到队列有空间
};

// 线程安全队列
class ThreadSaveQueue {

public:
    bool Push(const std::shared_ptr<BaseData> &data) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_list.size() > m_max_number) {
            switch (m_buffer_strategy) {
                case BufferOverStrategy::DROP_EARLY: {
                    // 缓存队列满了，丢弃最早的帧，保证实时性，但不丢弃其他信息数据
                    if (m_list.front()->get_data_type() == BaseDataType::ZJV_DATATYPE_FRAME) {
                        m_list.pop_front();
                        m_list.push_back(data);
                        m_work_cond->notify_one();
                        return false;  // 返回的false标识缓冲队列满
                    }
                    break;
                }
                case BufferOverStrategy::DROP_LATE: {
                    if (m_list.front()->get_data_type() == BaseDataType::ZJV_DATATYPE_FRAME) {
                        m_list.pop_back();
                        m_list.push_back(data);
                        m_work_cond->notify_one();
                        return false;  // 丢弃的是最新的帧
                    }
                    break;
                }
                case BufferOverStrategy::CLEAR: {
                    m_list.clear();
                    m_list.push_back(data);
                    m_work_cond->notify_one();
                    return false;
                }
                case BufferOverStrategy::BLOCK: {
                    m_self_cond.wait(lock);
                    break;
                }
                default: {
                    throw std::runtime_error("unknown buffer over strategy");
                }
            }
        } else {
            m_list.push_back(data);
            m_work_cond->notify_one();
        }
        return true;
    }

    bool Pop(std::shared_ptr<BaseData> &data) {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_list.empty()) {
            return false;
        }
        data = m_list.front();
        m_list.pop_front();
        if (m_buffer_strategy == BufferOverStrategy::BLOCK) {
            m_self_cond.notify_one();
        }
        return true;
    }

    void push_front(const std::shared_ptr<BaseData> &data) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_list.push_front(data);
        m_work_cond->notify_one();
    }

    void set_max_size(const int size) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_max_number = size;
    }

    int size() {
        std::unique_lock<std::mutex> lock(m_mutex);
        return (int)m_list.size();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_list.clear();
    }

    void setCond(std::shared_ptr<std::condition_variable> &cond) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_work_cond = cond;
    }

    void set_buffer_strategy(BufferOverStrategy strategy) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_buffer_strategy = strategy;
    }

private:
    std::mutex                                  m_mutex;
    std::shared_ptr<std::condition_variable>    m_work_cond;  // 用于唤醒工作线程的条件变量
    std::condition_variable                     m_self_cond;            // 用于唤醒自身的条件变量
    std::list<std::shared_ptr<BaseData> >       m_list;          // 缓冲队列
    int                                         m_max_number = 25;        // 默认最大缓冲帧数
    BufferOverStrategy                          m_buffer_strategy = ZJV_QUEUE_DROP_EARLY;  // 缓冲队列满时的策略
}; // class ThreadSaveQueue

} // namespace ZJVIDEO

#endif  // ZJVIDEO_THREADSAVEQUEUE_H
