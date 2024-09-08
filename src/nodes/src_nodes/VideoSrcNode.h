#ifndef ZJVIDEO_VIDEOSRCNODE_H
#define ZJVIDEO_VIDEOSRCNODE_H

#include "nodes/BaseNode.h"

#ifdef Enable_FFMPEG
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}
#endif

namespace ZJVIDEO
{

    enum VIDEO_TYPE
    {
        ZJV_VIDEO_TYPE_UNKNOWN = 0,
        ZJV_VIDEO_TYPE_FILE,
        ZJV_VIDEO_TYPE_CAMERA,
        ZJV_VIDEO_TYPE_RTSP,
        ZJV_VIDEO_TYPE_RTMP,
        ZJV_VIDEO_TYPE_HTTP,
        ZJV_VIDEO_TYPE_HLS

    }; // enum VIDEO_TYPE

    class VideoSrcNode : public BaseNode
    {

    public:
        VideoSrcNode(const NodeParam &param);
        virtual ~VideoSrcNode();
        VideoSrcNode() = delete;

    protected:
        virtual int worker();
        // parse,解析配置文件
        virtual int parse_configure(std::string cfg_file);
        // 根据配置文件， 初始化对象,输入输出队列
        virtual int init();
        virtual int process(const std::vector<std::shared_ptr<FlowData>> &datas);
        virtual int control(std::shared_ptr<ControlData> &data) ;

    private:
        int reinit();
        int deinit();

    private:
        int m_video_type; // 视频类型
        std::string m_video_path; // 视频路径
        bool m_initialed = false;

        int m_video_index; // 视频流索引
        int m_frame_cnt; // 帧计数

#ifdef Enable_FFMPEG   
        struct SwsContext* m_swsCtx; // 转换上下文     
        AVFrame *m_Frame; // 存储解码后的原始帧
        AVFrame* m_rgbFrame; // 存储转换后的RGB帧
        AVCodecContext *m_CodecCtx; // 编解码器上下文
        AVFormatContext *m_FormatCtx; // 格式上下文
#endif

    }; // class VideoSrcNode

} // namespace ZJVIDEO

#endif