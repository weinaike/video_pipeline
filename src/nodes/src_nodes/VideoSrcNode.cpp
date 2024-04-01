

#include "VideoSrcNode.h"
#include "nlohmann/json.hpp"

#define VIDEOSRC_LOG "VideoSrcNode"

namespace ZJVIDEO
{

    VideoSrcNode::VideoSrcNode(const NodeParam &param) : BaseNode(param)
    {
        m_logger = el::Loggers::getLogger(VIDEOSRC_LOG);
        m_batch_process = false;
        m_frame_cnt = 0;

        m_pFrame = NULL;
        m_rgbFrame = NULL;
        m_pCodecCtx = NULL;
        m_pFormatCtx = NULL;
        m_swsCtx = NULL;

        m_video_index = -1;
        m_initialed  = false;

        CLOG(INFO, VIDEOSRC_LOG) << "VideoSrcNode::VideoSrcNode";
        parse_configure(param.m_cfg_file);
        init();
    }

    VideoSrcNode::~VideoSrcNode()
    {
        CLOG(INFO, VIDEOSRC_LOG) << "VideoSrcNode::~VideoSrcNode";

        av_frame_free(&m_pFrame);
        av_frame_free(&m_rgbFrame);
        sws_freeContext(m_swsCtx);
        avcodec_close(m_pCodecCtx);
        avformat_close_input(&m_pFormatCtx);
        if (m_rgbFrame) {
            av_freep(&m_rgbFrame->data[0]);
        }
    }

    int VideoSrcNode::parse_configure(std::string cfg_file)
    {
        std::ifstream i(cfg_file);
        if (i.is_open() == false)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "open cfg_file failed";
            m_video_type = ZJV_VIDEO_TYPE_UNKNOWN;
            m_video_path = "";
            return ZJV_STATUS_ERROR;
        }
        nlohmann::json j;
        i >> j;
        std::string type = j["video_type"];
        if (type == "file")
        {
            m_video_type = ZJV_VIDEO_TYPE_FILE;
        }
        else if (type == "camera")
        {
            m_video_type = ZJV_VIDEO_TYPE_CAMERA;
        }
        else if (type == "rtsp")
        {
            m_video_type = ZJV_VIDEO_TYPE_RTSP;
        }
        else if (type == "rtmp")
        {
            m_video_type = ZJV_VIDEO_TYPE_RTMP;
        }
        else if (type == "http")
        {
            m_video_type = ZJV_VIDEO_TYPE_HTTP;
        }
        else if (type == "hls")
        {
            m_video_type = ZJV_VIDEO_TYPE_HLS;
        }
        else
        {
            m_video_type = ZJV_VIDEO_TYPE_UNKNOWN;
        }
        m_video_path = j["path"];

        // 打印配置参数
        CLOG(INFO, VIDEOSRC_LOG) << "----------------video src node config-----------------";
        CLOG(INFO, VIDEOSRC_LOG) << "video_type:    [" << m_video_type << "]";
        CLOG(INFO, VIDEOSRC_LOG) << "video_path:     [" << m_video_path << "]";
        CLOG(INFO, VIDEOSRC_LOG) << "------------------------------------------------------";

        return 0;
    }

    int VideoSrcNode::init()
    {
        AVCodec *pCodec;
        AVDictionary *options = NULL;
        uint8_t *out_buffer;

        avformat_network_init();
        m_pFormatCtx = avformat_alloc_context();

        av_dict_set(&options, "buffer_size", "131072", 0);
        av_dict_set(&options, "max_delay", "500000", 0);

        if (avformat_open_input(&m_pFormatCtx, m_video_path.c_str(), NULL, &options) != 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Couldn't open input stream.";
            return ZJV_STATUS_ERROR;
        }

        if (avformat_find_stream_info(m_pFormatCtx, NULL) < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Couldn't find stream information.";
            return ZJV_STATUS_ERROR;
        }

        m_video_index = av_find_best_stream(m_pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (m_video_index < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Couldn't find a video stream.";
            return ZJV_STATUS_ERROR;
        }

        m_pCodecCtx = avcodec_alloc_context3(NULL);
        if (!m_pCodecCtx)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Could not allocate video codec context.";
            return ZJV_STATUS_ERROR;
        }

        if (avcodec_parameters_to_context(m_pCodecCtx, m_pFormatCtx->streams[m_video_index]->codecpar) < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Could not copy the stream parameters.";
            return ZJV_STATUS_ERROR;
        }

        pCodec = avcodec_find_decoder(m_pCodecCtx->codec_id);
        if (!pCodec)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Codec not found.";
            return ZJV_STATUS_ERROR;
        }

        if (avcodec_open2(m_pCodecCtx, pCodec, NULL) < 0)
        {
            CLOG(ERROR, VIDEOSRC_LOG) << "Could not open codec.";
            return ZJV_STATUS_ERROR;
        }

        m_pFrame = av_frame_alloc();

        if (!m_pFrame) {
            return -1; // Failed to allocate the frame
        }

        m_swsCtx = sws_getContext(m_pCodecCtx->width, m_pCodecCtx->height, m_pCodecCtx->pix_fmt,
             m_pCodecCtx->width, m_pCodecCtx->height, AV_PIX_FMT_RGB24, SWS_BILINEAR, NULL, NULL, NULL);
        m_rgbFrame = av_frame_alloc();
        uint8_t* buffer = (uint8_t*)av_malloc(av_image_get_buffer_size(AV_PIX_FMT_RGB24, m_pCodecCtx->width, 
            m_pCodecCtx->height, 1));
        av_image_fill_arrays(m_rgbFrame->data, m_rgbFrame->linesize, buffer, AV_PIX_FMT_RGB24, m_pCodecCtx->width, 
                m_pCodecCtx->height, 1);


        m_initialed = true;

        CLOG(INFO, VIDEOSRC_LOG) << "VideoSrcNode::init";
        return 0;
    }

    int VideoSrcNode::worker()
    {
        while (m_run)
        {
            if(m_initialed == false)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            std::vector<std::shared_ptr<FlowData>> datas;
            datas.clear();
            std::shared_ptr<FrameData> frame_data = std::make_shared<FrameData>();
            std::shared_ptr<FlowData> flow = std::make_shared<FlowData>(frame_data);
            flow->set_channel_id(m_nodeparam.m_channel_id);
            datas.push_back(flow);

            // 主处理
            process(datas);

            bool has_output = false;
            for(const auto & data : datas)
            {
                std::vector<std::string> tags;
                for (const auto & output_data : m_nodeparam.m_output_datas)
                {
                    std::string name = m_nodeparam.m_node_name + "."+ output_data;
                    tags.push_back(name);
                }
            
                has_output = data->has_extras(tags);
            }

            if(has_output)
            {
                send_output_data(datas);
            }
        }
        return ZJV_STATUS_OK;
    }

    int VideoSrcNode::process_single(const std::vector<std::shared_ptr<const BaseData>> &in_metas,
                                     std::vector<std::shared_ptr<BaseData>> &out_metas)
    {

        AVPacket packet;
        // Read frames from the file
        if (av_read_frame(m_pFormatCtx, &packet) >= 0) 
        {
            // Is this a packet from the video stream?
            if (packet.stream_index == m_video_index) 
            {
                // Send the packet to the decoder
                if (avcodec_send_packet(m_pCodecCtx, &packet) < 0) {
                    return -1; // Failed to send packet
                }

                // Receive the frames from the decoder
                while (avcodec_receive_frame(m_pCodecCtx, m_pFrame) >= 0)
                {
                    assert(m_rgbFrame->data != NULL);
                    // Convert the image from its native format to RGB
                    sws_scale(m_swsCtx, m_pFrame->data, m_pFrame->linesize, 0, 
                            m_pCodecCtx->height, m_rgbFrame->data, m_rgbFrame->linesize);

                            
                    int width = m_pCodecCtx->width;
                    int height = m_pCodecCtx->height;
                    int channels = 3;

                    std::shared_ptr<FrameData> frame_data = std::make_shared<FrameData>(width, height, channels);
                    frame_data->camera_id = m_nodeparam.m_channel_id;
                    frame_data->frame_id = m_frame_cnt;
                    frame_data->format = ZJV_IMAGEFORMAT_RGB24;

                    unsigned char *data = (unsigned char *)frame_data->data->mutable_cpu_data();

                    memcpy(data, m_rgbFrame->data[0], width * height * channels);

                    out_metas.push_back(frame_data);
                    m_frame_cnt++;
                }
            }            
        }
        av_packet_unref(&packet);
        return 0;
    }

    REGISTER_NODE_CLASS(VideoSrc)

} // namespace ZJVIDEO
