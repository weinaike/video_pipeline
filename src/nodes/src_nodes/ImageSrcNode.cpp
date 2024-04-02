

#include "ImageSrcNode.h"
#define IMAGESRC_LOG "ImageSrcNode"
namespace ZJVIDEO {

ImageSrcNode::ImageSrcNode(const NodeParam & param) : BaseNode(param)
{
    m_logger = el::Loggers::getLogger(IMAGESRC_LOG);
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::ImageSrcNode";
}

ImageSrcNode::~ImageSrcNode()
{
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::~ImageSrcNode";
}

int ImageSrcNode::parse_configure(std::string cfg_file)
{
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::parse_configure";
    return 0;
}

int ImageSrcNode::init()
{
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::init";
    return 0;
}

int ImageSrcNode::process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas)
{
    for(const auto & in :in_metas)
    {
        if (in->data_name == "Frame")
        {
            std::shared_ptr<const FrameData> frame_data = std::dynamic_pointer_cast<const FrameData>(in);

            std::shared_ptr<FrameData> cvt ;
            //深度拷贝一份FrameData
            if(frame_data->format == ZJV_IMAGEFORMAT_RGB24 )
            {                            
                // 输入图像格式转换
                cvt = std::make_shared<FrameData>(*frame_data);
            }
            else if(frame_data->format == ZJV_IMAGEFORMAT_BGR24 )
            {
                cvt = std::make_shared<FrameData>(*frame_data);
            }
            else if(frame_data->format == ZJV_IMAGEFORMAT_YUV420P )
            {
                int w = frame_data->width;
                int h = frame_data->height;
                int s = frame_data->stride;
                cvt = std::make_shared<FrameData>(w, h, ZJV_IMAGEFORMAT_RGBP);
                cvt->camera_id = frame_data->camera_id;
                cvt->frame_id = frame_data->frame_id;
                cvt->pts = frame_data->pts;
                cvt->fps = frame_data->fps;

                int stride = cvt->stride;

                // YUV420P转RGBP
                int y_size = s * h;
                int u_size = y_size / 4;
                int v_size = y_size / 4;
                unsigned char *y = (unsigned char *)frame_data->data->cpu_data();
                unsigned char *u = y + y_size;
                unsigned char *v = u + u_size;

                unsigned char *rgb = (unsigned char *)cvt->data->mutable_cpu_data();
                int r_size = stride * h ;

                unsigned char *r_plane = rgb;
                unsigned char *g_plane = rgb + r_size;
                unsigned char *b_plane = rgb + 2 * r_size;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        int index = i * w + j;
                        int y_index = i * w + j;
                        int u_index = (i / 2) * (w / 2) + (j / 2);
                        int v_index = (i / 2) * (w / 2) + (j / 2);

                        int r = y[y_index] + 1.402 * (v[v_index] - 128);
                        int g = y[y_index] - 0.344136 * (u[u_index] - 128) - 0.714136 * (v[v_index] - 128);
                        int b = y[y_index] + 1.772 * (u[u_index] - 128);

                        r = r > 255 ? 255 : r;
                        r = r < 0 ? 0 : r;
                        g = g > 255 ? 255 : g;
                        g = g < 0 ? 0 : g;
                        b = b > 255 ? 255 : b;
                        b = b < 0 ? 0 : b;

                        r_plane[index] = r;
                        g_plane[index] = g;
                        b_plane[index] = b;
                    }
                }           
            }
            else if(frame_data->format == ZJV_IMAGEFORMAT_YUV422P )
            {
                // YUV422P转RGBP
                int w = frame_data->width;
                int h = frame_data->height;
                int s = frame_data->stride;
                cvt = std::make_shared<FrameData>(w, h, ZJV_IMAGEFORMAT_RGBP);

                cvt->camera_id = frame_data->camera_id;
                cvt->frame_id = frame_data->frame_id;
                cvt->pts = frame_data->pts;
                cvt->fps = frame_data->fps;

                int stride = cvt->stride;

                int y_size = s * h;
                int u_size = y_size / 2;
                int v_size = y_size / 2;
                unsigned char *y = (unsigned char *)frame_data->data->cpu_data();
                unsigned char *u = y + y_size;
                unsigned char *v = u + u_size;

                unsigned char *rgb = (unsigned char *)cvt->data->mutable_cpu_data();
                int r_size =  stride * h ;

                unsigned char *r_plane = rgb;
                unsigned char *g_plane = rgb + r_size;
                unsigned char *b_plane = rgb + 2 * r_size;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        int index = i * w + j;
                        int y_index = i * w + j;
                        int u_index = (i / 2) * (w / 2) + (j / 2);
                        int v_index = (i / 2) * (w / 2) + (j / 2);

                        int r = y[y_index] + 1.402 * (v[v_index] - 128);
                        int g = y[y_index] - 0.344136 * (u[u_index] - 128) - 0.714136 * (v[v_index] - 128);
                        int b = y[y_index] + 1.772 * (u[u_index] - 128);

                        r = r > 255 ? 255 : r;
                        r = r < 0 ? 0 : r;
                        g = g > 255 ? 255 : g;
                        g = g < 0 ? 0 : g;
                        b = b > 255 ? 255 : b;
                        b = b < 0 ? 0 : b;

                        r_plane[index] = r;
                        g_plane[index] = g;
                        b_plane[index] = b;
                    }
                }

            }
            else if(frame_data->format == ZJV_IMAGEFORMAT_YUV444P )
            {
                // YUV444P转RGBP
                int w = frame_data->width;
                int h = frame_data->height;
                int s = frame_data->stride;
                cvt = std::make_shared<FrameData>(w, h, ZJV_IMAGEFORMAT_RGBP);

                cvt->camera_id = frame_data->camera_id;
                cvt->frame_id = frame_data->frame_id;
                cvt->pts = frame_data->pts;
                cvt->fps = frame_data->fps;
                
                int stride = cvt->stride;

                int y_size = s * h;
                int u_size = y_size;
                int v_size = y_size;
                unsigned char *y = (unsigned char *)frame_data->data->cpu_data();
                unsigned char *u = y + y_size;
                unsigned char *v = u + u_size;

                unsigned char *rgb = (unsigned char *)cvt->data->mutable_cpu_data();
                int r_size =  stride * h ;

                unsigned char *r_plane = rgb;
                unsigned char *g_plane = rgb + r_size;
                unsigned char *b_plane = rgb + 2 * r_size;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        int index = i * w + j;
                        int y_index = i * w + j;
                        int u_index = i * w + j;
                        int v_index = i * w + j;

                        int r = y[y_index] + 1.402 * (v[v_index] - 128);
                        int g = y[y_index] - 0.344136 * (u[u_index] - 128) - 0.714136 * (v[v_index] - 128);
                        int b = y[y_index] + 1.772 * (u[u_index] - 128);

                        r = r > 255 ? 255 : r;
                        r = r < 0 ? 0 : r;
                        g = g > 255 ? 255 : g;
                        g = g < 0 ? 0 : g;
                        b = b > 255 ? 255 : b;
                        b = b < 0 ? 0 : b;

                        r_plane[index] = r;
                        g_plane[index] = g;
                        b_plane[index] = b;
                    }
                }               
            }
            else if(frame_data->format == ZJV_IMAGEFORMAT_NV21)
            {
                // NV21转RGBP
                int w = frame_data->width;
                int h = frame_data->height;
                int s = frame_data->stride;
                
                cvt = std::make_shared<FrameData>(w, h, ZJV_IMAGEFORMAT_RGBP);

                cvt->camera_id = frame_data->camera_id;
                cvt->frame_id = frame_data->frame_id;
                cvt->pts = frame_data->pts;
                cvt->fps = frame_data->fps;
                
                int stride = cvt->stride;

                int y_size = s * h;
                int uv_size = y_size / 2;
                unsigned char *y = (unsigned char *)frame_data->data->cpu_data();
                unsigned char *uv = y + y_size;

                unsigned char *rgb = (unsigned char *)cvt->data->mutable_cpu_data();
                int r_size =  stride * h ;

                unsigned char *r_plane = rgb;
                unsigned char *g_plane = rgb + r_size;
                unsigned char *b_plane = rgb + 2 * r_size;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        int index = i * w + j;
                        int y_index = i * w + j;
                        int uv_index = (i / 2) * w + (j / 2);

                        int r = y[y_index] + 1.402 * (uv[uv_index * 2] - 128);
                        int g = y[y_index] - 0.344136 * (uv[uv_index * 2 + 1] - 128) - 0.714136 * (uv[uv_index * 2] - 128);
                        int b = y[y_index] + 1.772 * (uv[uv_index * 2 + 1] - 128);


                        r = r > 255 ? 255 : r;
                        r = r < 0 ? 0 : r;
                        g = g > 255 ? 255 : g;
                        g = g < 0 ? 0 : g;
                        b = b > 255 ? 255 : b;
                        b = b < 0 ? 0 : b;

                        r_plane[index] = r;
                        g_plane[index] = g;
                        b_plane[index] = b;
                    }
                }
            }
            else if(frame_data->format == ZJV_IMAGEFORMAT_NV12)
            {
                // NV12转RGBP
                int w = frame_data->width;
                int h = frame_data->height;
                int s = frame_data->stride;
                
                cvt = std::make_shared<FrameData>(w, h, ZJV_IMAGEFORMAT_RGBP);
                cvt->camera_id = frame_data->camera_id;
                cvt->frame_id = frame_data->frame_id;
                cvt->pts = frame_data->pts;
                cvt->fps = frame_data->fps;
                
                int stride = cvt->stride;

                int y_size = s * h;
                int uv_size = y_size / 2;
                unsigned char *y = (unsigned char *)frame_data->data->cpu_data();
                unsigned char *uv = y + y_size;

                unsigned char *rgb = (unsigned char *)cvt->data->mutable_cpu_data();
                int r_size =  stride * h ;

                unsigned char *r_plane = rgb;
                unsigned char *g_plane = rgb + r_size;
                unsigned char *b_plane = rgb + 2 * r_size;

                for (int i = 0; i < h; i++)
                {
                    for (int j = 0; j < w; j++)
                    {
                        int index = i * w + j;
                        int y_index = i * w + j;
                        int uv_index = (i / 2) * w + (j / 2);

                        int r = y[y_index] + 1.402 * (uv[uv_index * 2 + 1] - 128);
                        int g = y[y_index] - 0.344136 * (uv[uv_index * 2] - 128) - 0.714136 * (uv[uv_index * 2 + 1] - 128);
                        int b = y[y_index] + 1.772 * (uv[uv_index * 2] - 128);

                        r = r > 255 ? 255 : r;
                        r = r < 0 ? 0 : r;
                        g = g > 255 ? 255 : g;
                        g = g < 0 ? 0 : g;
                        b = b > 255 ? 255 : b;
                        b = b < 0 ? 0 : b;

                        r_plane[index] = r;
                        g_plane[index] = g;
                        b_plane[index] = b;
                    }
                }
            }
            else
            {
                CLOG(ERROR, IMAGESRC_LOG) << "ImageSrcNode::process_single, unsupport format";
            }
            

            out_metas.push_back(cvt);
        }
        
    }
    // std::this_thread::sleep_for(std::chrono::milliseconds(40));

    // CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::process_single";
    return 0;
}

REGISTER_NODE_CLASS(ImageSrc)

} // namespace ZJVIDEO
