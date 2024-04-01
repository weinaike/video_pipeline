#include "BaseNode.h"
#include <assert.h>

namespace ZJVIDEO{

#define BASENODE_LOG "BaseNode"

BaseNode::BaseNode(const NodeParam & param) : AbstractNode(param),
    m_nodeparam(param)
{
    // 注册日志记录器
    m_logger = el::Loggers::getLogger(BASENODE_LOG);
    el::Configurations conf;
    conf.setToDefault();
    // Get the format for Info level
    std::string infoFormat = conf.get(el::Level::Info, el::ConfigurationType::Format)->value();
    // Set the format for Debug level to be the same as Info level
    conf.set(el::Level::Debug, el::ConfigurationType::Format, infoFormat);
    el::Loggers::reconfigureLogger(m_logger, conf);



    m_batch_process = param.m_channels;
    // 支持的最大批处理数量设置，这个应该直接从节点信息中获取，临时固定赋值。
    if(m_batch_process)
    {
        m_max_batch_size = 8;
    }
    if(m_batch_process)
    {
        for(const auto & item : m_nodeparam.m_input_node_datas)
        {
            int channel_id = parse_id(item.first);
            if(m_input_data_names_batch.find(channel_id)==m_input_data_names_batch.end())
            {
                m_input_data_names_batch[channel_id] = std::vector<std::string>();
                m_input_data_names_batch[channel_id].push_back(item.first + "." + item.second);        
            }
            else
            {
                // 通道id已经存在
                m_input_data_names_batch[channel_id].push_back(item.first + "." + item.second);
            }
        }

        for(const auto & item : m_input_data_names_batch[-1])
        {
            for(auto & a : m_input_data_names_batch)
            {
                if(a.first != -1)
                {
                    a.second.push_back(item);
                }                
            }
        }       
    }
    else
    {
        for (const auto & item : m_nodeparam.m_input_node_datas) 
        {
            std::string extra_nmae= item.first + "." + item.second;
            m_input_data_names.push_back(extra_nmae);
        }
    }



    // 输出节点信息
    CLOG(INFO, BASENODE_LOG) << "--------------" << m_nodeparam.m_node_name<<" create info--------------------";
    CLOG(INFO, BASENODE_LOG) << "node name: " << m_nodeparam.m_node_name;
    CLOG(INFO, BASENODE_LOG) << "node type: " << m_nodeparam.m_node_type;
    CLOG(INFO, BASENODE_LOG) << "node channels: " << m_nodeparam.m_channels;
    CLOG(INFO, BASENODE_LOG) << "node cfg file: " << m_nodeparam.m_cfg_file;
    CLOG(INFO, BASENODE_LOG) << "node channel id: " << m_nodeparam.m_channel_id;
    CLOG(INFO, BASENODE_LOG) << "node output datas: " << m_nodeparam.m_output_datas.size();
    for (const auto & item : m_nodeparam.m_output_datas) 
    {
        CLOG(INFO, BASENODE_LOG) << "       output data: " << item;
    }
    CLOG(INFO, BASENODE_LOG) << "node input node datas: " << m_nodeparam.m_input_node_datas.size();
    for (const auto & item : m_nodeparam.m_input_node_datas) 
    {
        CLOG(INFO, BASENODE_LOG) << "       input node data: " << item.first << " " << item.second;
    }
    CLOG(INFO, BASENODE_LOG) << "node batch process: " << m_nodeparam.m_channels;



    parse_configure(m_nodeparam.m_cfg_file);
    init();
}

BaseNode::~BaseNode()
{
    stop();
}

int BaseNode::parse_configure(std::string cfg_file)
{
    return ZJV_STATUS_OK;
} 

int BaseNode::init()
{
    
    return ZJV_STATUS_OK;
}

int BaseNode::connect_add_input(const std::string & tag, std::shared_ptr<FlowQueue> queue) 
{

    std::unique_lock<std::mutex> lk(m_base_mutex);
    queue->setCond(m_base_cond);
    m_input_buffers.insert(make_pair(tag, queue));

    return ZJV_STATUS_OK;
}

int BaseNode::connect_add_output(const std::string & tag, std::shared_ptr<FlowQueue> queue)
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    m_output_buffers.insert(make_pair(tag, queue));

    return ZJV_STATUS_OK;
}


int BaseNode::disconnect_del_input(const std::string & tag)
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_input_buffers.find(tag) != m_input_buffers.end()) 
    {
        m_input_buffers.erase(tag);
    }
    return ZJV_STATUS_OK;
}

int BaseNode::disconnect_del_output(const std::string & tag)
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_output_buffers.find(tag) != m_output_buffers.end()) 
    {
        m_output_buffers.erase(tag);
    }
    return ZJV_STATUS_OK;
}

int BaseNode::start()
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if (m_run) 
    {
        CLOG(INFO, BASENODE_LOG)<< m_nodeparam.m_node_name <<" thread is started already";
        return ZJV_STATUS_OK;
    }
    if(m_input_buffers.size() == 0 || m_output_buffers.size() == 0)
    {
        CLOG(ERROR, BASENODE_LOG)<<m_nodeparam.m_node_name << " not set input or output queue";
        return ZJV_STATUS_ERROR;
    }
    else
    {
        
        CLOG(INFO, BASENODE_LOG) <<  m_nodeparam.m_node_name << " input queue size :" << m_input_buffers.size() ;
        // 打印队列名称
        for(const auto & input :m_input_buffers)
        {
            CLOG(INFO, BASENODE_LOG)<<"        "<<input.first<<" ptr: "<<input.second.get();
        }
        CLOG(INFO, BASENODE_LOG) <<  m_nodeparam.m_node_name << " output queue size :" << m_output_buffers.size() ;
        for(const auto & output :m_output_buffers)
        {
            CLOG(INFO, BASENODE_LOG)<<"        "<<output.first<<" ptr: "<<output.second.get();
        }
    }

    // 节点输入需求，与输入队列是否已准备就绪
    if(m_nodeparam.m_input_node_datas.size() == 0)
    {
        m_node_position_type = ZJV_NODE_POSITION_SRC;
    }
    for(const auto & output : m_output_buffers)
    {
        if(output.first == m_nodeparam.m_node_name)
        {
             m_node_position_type = ZJV_NODE_POSITION_DST;
        }
    }

    for(const auto & item : m_nodeparam.m_input_node_datas)
    {
        if (m_input_buffers.find(item.first) == m_input_buffers.end()) 
        {
            CLOG(ERROR, BASENODE_LOG) << "input queue ["<< item.first <<"] is not ready in " << m_nodeparam.m_node_name ;
            return ZJV_STATUS_ERROR;
        }
    }
    if(m_node_position_type == ZJV_NODE_POSITION_UNKNOWN)
    {
        m_node_position_type = ZJV_NODE_POSITION_MID;
    }

    CLOG(INFO, BASENODE_LOG) <<  m_nodeparam.m_node_name << " type :" << m_node_position_type ;


    m_run = true;
    m_worker = std::thread(&BaseNode::worker, this);
    
    CLOG(INFO, BASENODE_LOG)<< m_nodeparam.m_node_name <<" thread start";
    return ZJV_STATUS_OK;
}


int BaseNode::stop()
{
    if (!m_run) 
    {
        CLOG(INFO, BASENODE_LOG)<< m_nodeparam.m_node_name << " thread exited already";
        return ZJV_STATUS_OK;
    }
    
    m_run = false;
    m_base_cond->notify_all();

    

    // 清空队列
    for (auto & it : m_input_buffers) 
    {
        it.second->clear();
    }
    for (auto & it : m_output_buffers) 
    {
        it.second->clear();
    }

    m_base_cond->notify_all();
    if (m_worker.joinable()) 
    {
        m_worker.join();
    }
    CLOG(INFO, BASENODE_LOG)<< m_nodeparam.m_node_name <<" thread exit success";
    return ZJV_STATUS_OK;
}


// 收集1各通道的一组数据（该组数据包含所有输入需求）
int BaseNode::get_input_data(std::vector<std::shared_ptr< FlowData>> &datas)
{
    int cnt = 0;

    // 单输入队列
    if(m_input_buffers.size() == 1)
    {
        // std::cout<<m_nodeparam.m_node_name << " single input queue"<<std::endl;
        for (const auto & buffer :m_input_buffers)
        {
            std::shared_ptr<FlowData> data;
            if(buffer.second->Pop(data))
            {
                datas.push_back(data);
                cnt++;
                if(cnt >= m_max_batch_size)
                {
                    break;
                }
            }
            else
            {
                // CLOG(INFO,BASENODE_LOG) <<m_nodeparam.m_node_name << "queue empty";
                break;
            }
        }
        
    }
    else // 多输入情况，先遍历队列，均摊负载
    {
        // std::cout<<m_nodeparam.m_node_name << " multi input queue"<<std::endl;
        if(m_nodeparam.m_channels == false)
        {
            assert(m_max_batch_size == 1);
        }
        // 单通道节点，意味着输入队列只有本通道的数据，其他通道的数据不会送入
        // 若出现其他通道数据输入，则是异常情况。单节点通道，每次仅处理一个数据
        // 每个队列中，的flowdata，都可以包含所有数据，存在冗余
        // 1. 从多个输入队列中，遭到最早的帧
        //      1.1 如果帧满足输入条件，则压入容器， 从其他输入队列中，去掉同一帧数据，避免重复计算
        //      1.2 如果帧不满足条件， 直接丢弃
        // 重复以上步骤，直到找到一个满足条件的最早的帧，
        int cnt = 0;
        while(cnt < m_max_batch_size)
        {
            // 判断那个队列，最前面的元素的时刻最早
            std::chrono::system_clock::time_point earliest_time = std::chrono::system_clock::now();
            std::string earliest_queue = "";
            for (const auto & buffer :m_input_buffers)
            {
                if (buffer.second->size() > 0) 
                {
                    // earliest_queue = buffer.first;
                    std::shared_ptr<FlowData> data = nullptr;
                    buffer.second->front(data);
                    auto time = data->create_time;

                    if (time < earliest_time) 
                    {
                        earliest_time = time;
                        earliest_queue = buffer.first;
                    }
                }
            }
            
            // 判断
            if (m_input_buffers.find(earliest_queue) == m_input_buffers.end()) 
            {
                return ZJV_STATUS_QUEUE_EMPTY;
            }
            else
            {   
                std::shared_ptr<FlowData> data = nullptr;
                m_input_buffers[earliest_queue]->front(data);
                // data->debug();
                // std::cout<<m_nodeparam.m_node_name <<  m_input_data_names.size()<<std::endl;
                // for(const auto & a :m_input_data_names)
                // {
                //     std::cout << m_nodeparam.m_node_name <<  a <<std::endl;
                // }
                if(m_batch_process)
                {
                    m_input_data_names = m_input_data_names_batch[data->get_channel_id()];
                }
                if(data->has_extras(m_input_data_names))
                {
                    // std::cout<<m_nodeparam.m_node_name<<" data ready"<<std::endl;
                    m_input_buffers[earliest_queue]->Pop(data);
                    // data.get()已处理队列中
                    bool repeat = false;
                    for(const auto & item : m_dealed_smaple)
                    {
                        if(data.get() == item)
                        {
                            // std::cout<<m_nodeparam.m_node_name<<" repeat data " << data.get()<<std::endl;
                            repeat = true;
                            continue;
                        }
                    }
                    if(!repeat)
                    {
                        datas.push_back(data);
                        cnt++;
                        if(m_dealed_smaple.size()>=64)
                        {
                            m_dealed_smaple.pop_front(); 
                            m_dealed_smaple.push_back(data.get());                       
                        }
                        else
                        {
                            m_dealed_smaple.push_back(data.get());
                        }
                    } 

                    
                    for (const auto & buffer :m_input_buffers)
                    {
                        if(earliest_queue == buffer.first)
                        {
                            continue;
                        }
                        std::shared_ptr<FlowData> temp = nullptr;
                        if (buffer.second->front(temp)) 
                        {                            
                            if (data.get() ==  temp.get())  // 指向同一对象
                            {
                                // std::cout<<m_nodeparam.m_node_name<<" repeat data " << data.get()<<std::endl;
                                buffer.second->Pop(temp);
                            }
                        }
                    }
                }
                else
                {
                    // std::cout<<m_nodeparam.m_node_name<<" data not ready"<<std::endl;
                    m_input_buffers[earliest_queue]->Pop(data);
                }                
            }
        }
    }

    return ZJV_STATUS_OK;
}
int BaseNode::send_output_data(const std::vector<std::shared_ptr<FlowData>> &datas)
{
    if(m_output_buffers.size() == 1)
    {
        // 仅一个输出队列，直接发送
        for (const auto & item : m_output_buffers) 
        {
            for (const auto & data : datas) 
            {
                item.second->Push(data);
            }
        }
    }
    else // 对于多输出队列
    {
        for (const auto & item : m_output_buffers) 
        {
            int id = parse_id(item.first);
            // 通过解析id，判断后续节点是否支撑多通道，因为已经经过扩展了,会带通道信息。
            // 如果经过扩展，则需要按通道分发，若未扩展，则直接发送
            if(id < 0) 
            {
                for (const auto & data : datas) 
                {
                    item.second->Push(data);
                    // std::cout<< m_nodeparam.m_node_name << " " << item.first<<std::endl;
                    // data->debug();
                }
            }
            else
            {
                // 发送到对应管道
                for (const auto & data : datas) 
                {
                    if(id == data->get_channel_id())
                    {
                        item.second->Push(data);
                        // std::cout<< m_nodeparam.m_node_name << " " << item.first<<std::endl;
                        // data->debug();
                    }   
                } 
            }
        }
    }




    return ZJV_STATUS_OK;

}


int BaseNode::worker()
{    
    while (m_run) 
    {
        std::vector<std::shared_ptr<FlowData>> datas;
        datas.clear();

        get_input_data(datas);

        if (datas.size() == 0) 
        {
            // std::cout<<m_nodeparam.m_node_name <<" worker is wait "<<std::endl;
            std::unique_lock<std::mutex> lk(m_base_mutex);
            m_base_cond->wait(lk);
            // std::cout<<m_nodeparam.m_node_name <<" worker wait is notified"<<std::endl;
            continue;
        }
        else
        {
            // 主处理
            process(datas);
            // 传递数据
            send_output_data(datas);
        }         
    }
    return ZJV_STATUS_OK;
}

int BaseNode::process_batch(const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                    std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch)
{
    // 根据需要批量处理
    for(const auto & in_metas : in_metas_batch)
    {
        std::vector<std::shared_ptr<BaseData> > out_metas;
        process_single(in_metas, out_metas);
        out_metas_batch.push_back(out_metas);
    }
    return ZJV_STATUS_OK;
}


int BaseNode::process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                    std::vector<std::shared_ptr<BaseData> > & out_metas)

{
    // 仅供调试使用， 每个示例都需要重写该代码， 根据输出类型明确构建输出数据
    for (const auto & output_data : m_nodeparam.m_output_datas)
    {
        std::shared_ptr<ExtraData> data = std::make_shared<ExtraData>();
        data->data_name = output_data;
        out_metas.push_back(data);
        // std::cout<<__LINE__<<output_data<<std::endl;
    }
    return ZJV_STATUS_OK;
}


int BaseNode::process(const std::vector<std::shared_ptr<FlowData>> & datas)
{
    // std::cout<<m_nodeparam.m_node_name << " process , input size: " << datas.size()<<std::endl;

    auto start = std::chrono::system_clock::now();
    if(m_batch_process)
    {
        std::vector<std::vector<std::shared_ptr<const BaseData> >> in_metas_batch;
        for(const auto & data :datas)
        {
            std::vector<std::string > tags;
            int channel_id = data->get_channel_id();
            tags = m_input_data_names_batch[channel_id];
            std::vector<std::shared_ptr<const BaseData>> in_metas;
            if((tags.size()>0) )
            {
                if(data->get_extras(tags, in_metas) == ZJV_STATUS_ERROR)
                {
                    // data->debug();
                    std::cout<<m_nodeparam.m_node_name << " get_extras error"<<std::endl;
                    assert(0);
                }
            }

            in_metas_batch.push_back(in_metas);
        }

        std::vector<std::vector<std::shared_ptr<BaseData>>> out_metas_batch;
        
        process_batch(in_metas_batch, out_metas_batch);


        if(in_metas_batch.size() != out_metas_batch.size())
        {
            std::cout<< m_nodeparam.m_node_name << "in_metas_batch.size() " <<in_metas_batch.size() 
                << " out_metas_batch.size() " <<out_metas_batch.size()<<std::endl;
            assert(0);
        }


        for(int i = 0 ; i < datas.size(); i++)
        {
            const auto & out = out_metas_batch[i];
            std::vector<std::pair<std::string, std::shared_ptr<const BaseData> > > result;
            assert(out.size() == m_nodeparam.m_output_datas.size());

            for (const auto & output_data : m_nodeparam.m_output_datas)
            {
                bool not_found = true;
                for(const auto & out : out_metas_batch[i])
                {
                    if(out->data_name == output_data)
                    {
                        std::string name = m_nodeparam.m_node_name + "."+ output_data;
                        result.push_back({name, out});
                        // std::cout<<m_nodeparam.m_node_name<< name<<std::endl;
                        not_found = false;
                    }
                }
                if(not_found)
                {
                    CLOG(ERROR, BASENODE_LOG) << m_nodeparam.m_node_name << " output data ["<< output_data <<"] is not found in " << m_nodeparam.m_node_name ;
                    return ZJV_STATUS_ERROR;
                }
            }

            datas[i]->push_back(result);
        }
    }
    else
    {        
        for(const auto & data :datas)
        {            
            std::vector<std::string > tags;
            for (const auto & input_data : m_nodeparam.m_input_node_datas)
            {
                tags.push_back(input_data.first + "." + input_data.second);
            }
            std::vector<std::shared_ptr<const BaseData>> in_metas;
            if(tags.size() > 0)
            {
                if(data->get_extras(tags, in_metas) == ZJV_STATUS_ERROR)
                {
                    assert(0);
                }
            }
            else if(m_node_position_type == ZJV_NODE_POSITION_SRC) // 起始节点
            {
                in_metas.push_back(data->get_frame());
            }
            
            std::vector<std::shared_ptr<BaseData> > out_metas;

            process_single(in_metas, out_metas);    

            if(out_metas.size() != m_nodeparam.m_output_datas.size())
            {
                std::cout<<m_nodeparam.m_node_name<<" out_metas.size() "<<out_metas.size()<<std::endl;
                for(const auto & out :out_metas)
                {
                    std::cout<<out->data_name<<std::endl;
                }
                continue; // 输出不正确，直接跳过
            }
            // 输出
            std::vector<std::pair<std::string, std::shared_ptr<const BaseData> > > result;
            for (const auto & output_data : m_nodeparam.m_output_datas)
            {
                bool not_found = true;
                for(const auto & out : out_metas)
                {
                    if(out->data_name == output_data)
                    {
                        std::string name = m_nodeparam.m_node_name + "."+ output_data;
                        result.push_back({name, out});
                        // std::cout<<name<<std::endl;
                        not_found = false;
                    }
                }
                if(not_found)
                {
                    CLOG(ERROR, BASENODE_LOG) << "output data ["<< output_data <<"] is not found in " << m_nodeparam.m_node_name ;
                    return ZJV_STATUS_ERROR;
                }
            }
            data->push_back(result);
        }

    }

    auto end = std::chrono::system_clock::now();


    std::chrono::duration<double> elapsed_seconds = end - start; // Calculate elapsed time

    double fps = datas.size() / elapsed_seconds.count() ; 
    m_fps = m_fps*m_fps_count/(m_fps_count+1) + fps/(m_fps_count+1);
    m_fps_count++;
    if(m_fps_count > 100)
    {
        m_fps_count = 0;
    }
    return ZJV_STATUS_OK;
}

bool BaseNode::get_run_status()
{
    return m_run;
}
int BaseNode::get_control_info(std::shared_ptr<ControlData>& data ) 
{
    std::unique_lock<std::mutex> lk(m_base_mutex);
    if(data->get_control_type() == ZJV_CONTROLTYPE_GET_FPS)
    {
        std::shared_ptr<GetFPSControlData> ptr = std::dynamic_pointer_cast<GetFPSControlData>(data);
        ptr->set_fps(m_fps);
    }
    else if(data->get_control_type() == ZJV_CONTROLTYPE_SET_LOGGER_LEVEL)
    {
        std::shared_ptr<SetLoggerLevelControlData> ptr = std::dynamic_pointer_cast<SetLoggerLevelControlData>(data);
        int level = ptr->get_level();
        m_logger->configurations()->set(el::Level::Global, el::ConfigurationType::Enabled, "true");
        if(ZJV_LOGGER_LEVEL_INFO == level)
        {
            m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
        }
        else if(ZJV_LOGGER_LEVEL_WARN == level)
        {
            m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            m_logger->configurations()->set(el::Level::Info, el::ConfigurationType::Enabled, "false");
        }
        else if(ZJV_LOGGER_LEVEL_ERROR == level)
        {
            m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            m_logger->configurations()->set(el::Level::Info, el::ConfigurationType::Enabled, "false");
            m_logger->configurations()->set(el::Level::Warning, el::ConfigurationType::Enabled, "false");
        }
        else if(ZJV_LOGGER_LEVEL_FATAL == level)
        {
            m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            m_logger->configurations()->set(el::Level::Info, el::ConfigurationType::Enabled, "false");
            m_logger->configurations()->set(el::Level::Warning, el::ConfigurationType::Enabled, "false");
            m_logger->configurations()->set(el::Level::Error, el::ConfigurationType::Enabled, "false");
        }
        m_logger->reconfigure();
    }
    else
    {
        CLOG(ERROR, BASENODE_LOG) << "control type is not supported";
        return ZJV_STATUS_ERROR;
    
    }
    return ZJV_STATUS_OK;
}

std::string BaseNode::get_name()
{
    return m_nodeparam.m_node_name;
}

REGISTER_NODE_CLASS(Base)

} // namespace ZJVIDEO