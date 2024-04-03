

#include "VideoSrcNode.h"
#include "nlohmann/json.hpp"

#define VIDEOSRC_LOG "VideoSrcNode"

namespace ZJVIDEO
{

    VideoSrcNode::VideoSrcNode(const NodeParam &param) : BaseNode(param)
    {
        m_logger = el::Loggers::getLogger(VIDEOSRC_LOG);
        m_batch_process = false;
        m_max_batch_size = 1;
        m_frame_cnt = 0;

        m_Frame = NULL;
        m_rgbFrame = NULL;
        m_CodecCtx = NULL;
        m_FormatCtx = NULL;
        m_swsCtx = NULL;

        m_video_index = -1;
        m_initialed = false;
        m_video_path = "";

        CLOG(INFO, VIDEOSRC_LOG) << "VideoSrcNode::VideoSrcNode";
        parse_configure(param.m_cfg_file);
    }

    VideoSrcNode::~VideoSrcNode()
    {
        CLOG(INFO, VIDEOSRC_LOG) << "VideoSrcNode::~VideoSrcNode";
        deinit();
    }

    int VideoSrcNode::deinit()
    {
        av_frame_free(&m_Frame);
        av_frame_free(&m_rgbFrame);
        sws_freeContext(m_swsCtx);
        avcodec_close(m_CodecCtx);
        avformat_close_input(&m_FormatCtx);
        if (m_rgbFrame)
        {
            av_freep(&m_rgbFrame->data[0]);
        }
        m_initialed = false;
        return 0;
    }
    int VideoSrcNode::reinit()
    {
        deinit();
        init();
        return 0;
    }

    int VideoSrcNode::parse_configure(std::string cfg_file)
    {
        return 0;
    }

    int VideoSrcNode::init()
    {
        AVCodec *pCodec;
        AVDictionary *options = NULL;
        uint8_t *out_buffer;

        avformat_network_init();
        m_FormatCtx = avformat_alloc_context();

        av_dict_set(&options, "buffer_size", "131072", 0);
        av_dict_set(&options, "max_delay", "500000", 0);

        if (avformat_open_input(&m_FormatCtx, m_video_path.c_str(), NULL, &options) != 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Couldn't open input stream.";
            return ZJV_STATUS_ERROR;
        }

        if (avformat_find_stream_info(m_FormatCtx, NULL) < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Couldn't find stream information.";
            return ZJV_STATUS_ERROR;
        }

        m_video_index = av_find_best_stream(m_FormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (m_video_index < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Couldn't find a video stream.";
            return ZJV_STATUS_ERROR;
        }

        m_CodecCtx = avcodec_alloc_context3(NULL);
        if (!m_CodecCtx)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Could not allocate video codec context.";
            return ZJV_STATUS_ERROR;
        }

        if (avcodec_parameters_to_context(m_CodecCtx, m_FormatCtx->streams[m_video_index]->codecpar) < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Could not copy the stream parameters.";
            return ZJV_STATUS_ERROR;
        }

        pCodec = avcodec_find_decoder(m_CodecCtx->codec_id);
        if (!pCodec)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Codec not found.";
            return ZJV_STATUS_ERROR;
        }

        if (avcodec_open2(m_CodecCtx, pCodec, NULL) < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Could not open codec.";
            return ZJV_STATUS_ERROR;
        }

        m_Frame = av_frame_alloc();

        if (!m_Frame)
        {
            return -1; // Failed to allocate the frame
        }

        m_swsCtx = sws_getContext(m_CodecCtx->width, m_CodecCtx->height, m_CodecCtx->pix_fmt,
                                  m_CodecCtx->width, m_CodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
        m_rgbFrame = av_frame_alloc();
        uint8_t *buffer = (uint8_t *)av_malloc(av_image_get_buffer_size(AV_PIX_FMT_RGB24, m_CodecCtx->width,
                                                                        m_CodecCtx->height, 1));
        av_image_fill_arrays(m_rgbFrame->data, m_rgbFrame->linesize, buffer, AV_PIX_FMT_RGB24, m_CodecCtx->width,
                             m_CodecCtx->height, 1);

        m_initialed = true;

        CLOG(INFO, VIDEOSRC_LOG) << "VideoSrcNode::init";
        return 0;
    }

    int VideoSrcNode::worker()
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
                for (auto &data : datas)
                {
                    std::shared_ptr<const VideoData> video = data->get_video();

                    m_video_path = video->video_path;

                    reinit(); // 初始化

                    // 主处理
                    process(datas);
                    // 处理完当前视频， 再处理下一个
                }
            }
        }
        return ZJV_STATUS_OK;
    }

    int VideoSrcNode::process(const std::vector<std::shared_ptr<FlowData>> &datas)
    {

        AVPacket packet;
        // Read frames from the file
        while (av_read_frame(m_FormatCtx, &packet) >= 0 && m_run)
        {
            auto start = std::chrono::system_clock::now();
            // Is this a packet from the video stream?
            if (packet.stream_index == m_video_index)
            {
                // Send the packet to the decoder
                if (avcodec_send_packet(m_CodecCtx, &packet) < 0)
                {
                    return -1; // Failed to send packet
                }

                // Receive the frames from the decoder
                while (avcodec_receive_frame(m_CodecCtx, m_Frame) >= 0 && m_run)
                {
                    // Convert the image from its native format to RGB
                    sws_scale(m_swsCtx, m_Frame->data, m_Frame->linesize, 0,
                              m_CodecCtx->height, m_rgbFrame->data, m_rgbFrame->linesize);

                    int width = m_CodecCtx->width;
                    int height = m_CodecCtx->height;

                    std::shared_ptr<FrameData> frame_data = std::make_shared<FrameData>(width, height, ZJV_IMAGEFORMAT_RGB24);

                    frame_data->camera_id = m_nodeparam.m_channel_id;
                    frame_data->frame_id = m_frame_cnt;
                    frame_data->fps = m_CodecCtx->framerate.num / m_CodecCtx->framerate.den;
                    frame_data->pts = m_Frame->pts ;
                    // std::cout<<frame_data->stride<<std::endl;

                    unsigned char *data = (unsigned char *)frame_data->data->mutable_cpu_data();
                    memcpy(data, m_rgbFrame->data[0], width * height * 3);
                    m_frame_cnt++;

                    // 重新构造数据流
                    std::shared_ptr<FlowData> flow = std::make_shared<FlowData>(frame_data);
                    flow->set_channel_id(m_nodeparam.m_channel_id);

                    // 添加额外数据
                    std::vector<std::pair<std::string, std::shared_ptr<const BaseData>>> result;
                    bool not_found = true;
                    for (const auto &output_data : m_nodeparam.m_output_datas)
                    {
                        if (frame_data->data_name == output_data)
                        {
                            std::string name = m_nodeparam.m_node_name + "." + output_data;
                            result.push_back({name, frame_data});
                            // std::cout<<name<<std::endl;
                            not_found = false;
                        }
                    }
                    if (not_found)
                    {
                        CLOG(ERROR, VIDEOSRC_LOG) << "output data [" << frame_data->data_name << "] is not found in " << m_nodeparam.m_node_name;
                        break;
                    }
                    flow->push_back(result);

                    // 发送数据
                    std::vector<std::shared_ptr<FlowData>> flows;
                    flows.push_back(flow);
                    send_output_data(flows);
                    // std::cout<<frame_data->frame_id<<std::endl;
                }
            }
            av_packet_unref(&packet);
            // 延时
            // std::this_thread::sleep_for(std::chrono::milliseconds(30));

            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start; // Calculate elapsed time

            double fps = datas.size() / elapsed_seconds.count();
            m_fps = m_fps * m_fps_count / (m_fps_count + 1) + fps / (m_fps_count + 1);
            m_fps_count++;
            if (m_fps_count > 100)
            {
                m_fps_count = 0;
            }
        }

        return 0;
    }

    int VideoSrcNode::control(std::shared_ptr<ControlData> &data)
    {
        std::unique_lock<std::mutex> lk(m_base_mutex);
        if (data->get_control_type() == ZJV_CONTROLTYPE_SET_RUN_MODE)
        {
            std::shared_ptr<SetRunModeControlData> mode = std::dynamic_pointer_cast<SetRunModeControlData>(data);
            int run_mode = mode->get_mode();
            // 遍历 m_output_buffers
            for (auto &output : m_output_buffers)
            {
                if (run_mode == ZJV_PIPELINE_RUN_MODE_LIVING)
                {
                    output.second->set_buffer_strategy(BufferOverStrategy::ZJV_QUEUE_DROP_LATE);
                }
                else if (run_mode == ZJV_PIPELINE_RUN_MODE_RECORDED)
                {
                    output.second->set_buffer_strategy(BufferOverStrategy::ZJV_QUEUE_BLOCK);
                }
            }
            CLOG(INFO, VIDEOSRC_LOG) << "SetRunModeControlData::run_mode: " << run_mode;
        }
        else if (data->get_control_type() == ZJV_CONTROLTYPE_GET_FPS)
        {
            std::shared_ptr<GetFPSControlData> ptr = std::dynamic_pointer_cast<GetFPSControlData>(data);
            ptr->set_fps(m_fps);
        }
        else if (data->get_control_type() == ZJV_CONTROLTYPE_SET_LOGGER_LEVEL)
        {
            std::shared_ptr<SetLoggerLevelControlData> ptr = std::dynamic_pointer_cast<SetLoggerLevelControlData>(data);
            int level = ptr->get_level();
            m_logger->configurations()->set(el::Level::Global, el::ConfigurationType::Enabled, "true");
            if (ZJV_LOGGER_LEVEL_DEBUG == level)
            {
                m_logger->configurations()->set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
            }
            else if (ZJV_LOGGER_LEVEL_INFO == level)
            {
                m_logger->configurations()->set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
            }
            else if (ZJV_LOGGER_LEVEL_WARN == level)
            {
                m_logger->configurations()->set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Info, el::ConfigurationType::Enabled, "false");
            }
            else if (ZJV_LOGGER_LEVEL_ERROR == level)
            {
                m_logger->configurations()->set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Info, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Warning, el::ConfigurationType::Enabled, "false");
            }
            else if (ZJV_LOGGER_LEVEL_FATAL == level)
            {
                m_logger->configurations()->set(el::Level::Trace, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Info, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Warning, el::ConfigurationType::Enabled, "false");
                m_logger->configurations()->set(el::Level::Error, el::ConfigurationType::Enabled, "false");
            }
            m_logger->reconfigure();
        }
        return 0;
    }

    REGISTER_NODE_CLASS(VideoSrc)

} // namespace ZJVIDEO
