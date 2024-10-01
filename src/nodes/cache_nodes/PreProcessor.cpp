#include "PreProcessor.h"


#include "CImg/CImg.h"
#include "cimg_util.cpp"

using namespace cimg_library;

namespace ZJVIDEO
{
 

PreProcessor::PreProcessor(int lib_type, int device_id )
    : m_lib_type(lib_type), m_device_id(device_id)
{
    el::Loggers::getLogger(PRELOG);
}

int PreProcessor::parse_json(const nlohmann::json & j)
{
    m_param.output_name = j["output_name"];
    if (j.contains("param") ) 
    {
        try {
            m_param.letterbox_value = j["param"]["letterbox_color"].get<std::vector<int>>();
            m_param.do_normalize = j["param"]["normalize"];

            m_param.output_dims = j["param"]["output_diims"].get<std::vector<int>>();

            std::string output_format = j["param"]["output_format"];
            if(output_format == "NCHW")
            {
                m_param.output_format = ZJV_PREPROCESS_OUTPUT_FORMAT_NCHW;
                m_param.resize_channel = m_param.output_dims[1];
                m_param.resize_height = m_param.output_dims[2];
                m_param.resize_width = m_param.output_dims[3];
            }
            else if(output_format == "NHWC") 
            {
                m_param.output_format = ZJV_PREPROCESS_OUTPUT_FORMAT_NHWC;
                m_param.resize_height = m_param.output_dims[1];
                m_param.resize_width = m_param.output_dims[2];
                m_param.resize_channel = m_param.output_dims[3];
            }
            else if(output_format == "NCTHW")
            {
                m_param.output_format = ZJV_PREPROCESS_OUTPUT_FORMAT_NCTHW;
                m_param.resize_channel = m_param.output_dims[1];
                m_param.resize_height = m_param.output_dims[3];
                m_param.resize_width = m_param.output_dims[4];
            }
            else if (output_format == "NTCHW")
            {
                m_param.output_format = ZJV_PREPROCESS_OUTPUT_FORMAT_NTCHW;
                m_param.resize_channel = m_param.output_dims[2];
                m_param.resize_height = m_param.output_dims[3];
                m_param.resize_width = m_param.output_dims[4];
            }
            else if(output_format == "NTHW")
            {
                m_param.output_format = ZJV_PREPROCESS_OUTPUT_FORMAT_NTHW;
                m_param.resize_channel = 1;
                m_param.resize_height = m_param.output_dims[2];
                m_param.resize_width = m_param.output_dims[3];
            }
            else 
            {
                m_param.output_format = ZJV_PREPROCESS_OUTPUT_FORMAT_UNKNOWN;
                CLOG(ERROR, PRELOG)<<"output_format not supported now in " ;
                assert(0);
            }

            m_param.mean_value = j["param"]["mean"].get<std::vector<float>>();
            m_param.std_value = j["param"]["std"].get<std::vector<float>>();
            if( m_param.mean_value.size() != m_param.resize_channel 
                || m_param.std_value.size() != m_param.resize_channel)
            {
                CLOG(ERROR, PRELOG) << "mean or std size not match resize_channel in " ;
                assert(0);
            }

            std::string resize_type = j["param"]["resize_type"];
            if(resize_type == "Stretch") m_param.resize_type = ZJV_PREPROCESS_RESIZE_STRETCH;
            else if(resize_type == "LetterBox") m_param.resize_type = ZJV_PREPROCESS_RESIZE_LETTERBOX;
            else if(resize_type == "Fill") m_param.resize_type = ZJV_PREPROCESS_RESIZE_FILL;
            else m_param.resize_type = ZJV_PREPROCESS_RESIZE_UNKNOWN;

            if(m_param.resize_type == ZJV_PREPROCESS_RESIZE_UNKNOWN ) 
            {
                CLOG(ERROR, PRELOG)<<"resize_type not supported now in " ;
                assert(0);
            }

            std::string interp_type = j["param"]["interp_mode"];
            if(interp_type == "Linear") m_param.interp_type = ZJV_PREPROCESS_INTERP_LINEAR;
            else if(interp_type == "Nearest") m_param.interp_type = ZJV_PREPROCESS_INTERP_NEAREST;
            else if(interp_type == "Cubic") m_param.interp_type = ZJV_PREPROCESS_INTERP_CUBIC;
            else m_param.interp_type = ZJV_PREPROCESS_INTERP_UNKNOWN;

            if(m_param.interp_type == ZJV_PREPROCESS_INTERP_UNKNOWN ) 
            {
                CLOG(ERROR, PRELOG)<< "interp_type not supported now in " ;
                assert(0);
            }

            if(j["param"].contains("channel_format"))
            {
                std::string channel_format = j["param"]["channel_format"];
                if(channel_format == "RGB") m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_RGB;
                else if(channel_format == "BGR") m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_BGR;
                else m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_UNKNOWN;
            }

            // 打印预处理配置参数
            CLOG(INFO, PRELOG) << "------- preprocess config ------------";
            CLOG(INFO, PRELOG) << "resize_width   [" << m_param.resize_width << "]";
            CLOG(INFO, PRELOG) << "resize_height  [" << m_param.resize_height<<"]";
            CLOG(INFO, PRELOG) << "resize_channel [" << m_param.resize_channel<< "]";
            if(m_param.resize_channel == 3)
            {
                CLOG(INFO, PRELOG) << "mean:          [" << m_param.mean_value[0] << "," 
                    << m_param.mean_value[1] << "," << m_param.mean_value[2]<<"]";
                CLOG(INFO, PRELOG) << "std:           [" << m_param.std_value[0] << "," 
                    << m_param.std_value[1] << "," << m_param.std_value[2]<<"]";
                CLOG(INFO, PRELOG) << "letterbox:     [" << m_param.letterbox_value[0] << "," 
                    << m_param.letterbox_value[1] << "," << m_param.letterbox_value[2]<<"]";
            }
            else if(m_param.resize_channel == 1)
            {
                CLOG(INFO, PRELOG) << "mean:      [" << m_param.mean_value[0]<<"]";
                CLOG(INFO, PRELOG) << "std:       [" << m_param.std_value[0]<<"]";
            }
            else
            {
                CLOG(ERROR, PRELOG) << "resize_channel not supported now";
            }
            

            CLOG(INFO, PRELOG) << "do_normalize:  [" << m_param.do_normalize<<"]";
            CLOG(INFO, PRELOG) << "resize_type:   [" << m_param.resize_type<<"]";
            CLOG(INFO, PRELOG) << "interp_type:   [" << m_param.interp_type<<"]";
            CLOG(INFO, PRELOG) << "channel_format:[" << m_param.channel_format<<"]";
            CLOG(INFO, PRELOG) << "output_format: [" << m_param.output_format<<"]";     
            CLOG(INFO, PRELOG) << "------- preprocess config ------------";

        }
        catch (nlohmann::json::exception& e) {
            CLOG(ERROR, PRELOG) << "parse preprocess failed" << e.what();
        }
    }
    
    return ZJV_STATUS_OK;
}


int PreProcessor::run_cimg(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param)
{


    float * input_data = blob.mutable_cpu_data();
    assert(input_data != nullptr);
    int count = param.resize_channel * param.resize_height * param.resize_width;

    
    for(int i = 0; i < frame_rois.size(); i++)
    {        
        // 1. 提取图片
        const std::shared_ptr<FrameROI> frame_roi = frame_rois[i];
        std::shared_ptr<const FrameData> frame_data = frame_roi->frame;
        Rect roi = frame_roi->roi;
        CImg<float> img_float;
        cimg_preprocess(frame_data, roi, img_float, param);    

        if(img_float.size() != count)
        {
            CLOG(ERROR, PRELOG) << "img_float size not match count" << img_float.size() << "!=" << count;
            assert(img_float.size() == count);
        }

        std::memcpy(input_data + count * i, (float * )img_float.data(), img_float.size() * sizeof(float));
    }


    return ZJV_STATUS_OK;

}

int PreProcessor::run(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param)
{    
    for(int i = 0; i < frame_rois.size(); i++)
    {
        frame_rois[i]->pre_process = &param;
        frame_rois[i]->resize_type = param.resize_type;
        frame_rois[i]->input_width = param.resize_width;
        frame_rois[i]->input_height = param.resize_height;

        float scalex, scaley, padx, pady;
        get_scale_pad(frame_rois[i]->roi, param, scalex, scaley, padx, pady);
        frame_rois[i]->scale_x = scalex;
        frame_rois[i]->scale_y = scaley;
        frame_rois[i]->padx = padx;
        frame_rois[i]->pady = pady;
    }

    if(m_lib_type == ZJV_PREPROCESS_LIB_CIMG)
    {
        if(param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NCHW || param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NHWC)
        {
            return run_cimg(frame_rois, blob, param);
        }
        else if(param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NCTHW || param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NTCHW)
        {
            // todo for 3D model
        }
        else
        {
            CLOG(ERROR, PRELOG) << "output_format not supported now";
            assert(0);
        }
    }
    else if( m_lib_type == ZJV_PREPROCESS_LIB_CUDA)
    {
        // return run_cimg(frame_rois, blob, param);
        if (param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NCHW || param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NHWC)
        {
            return run_cuda(frame_rois, blob, param);
        }
        else if (param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NCTHW || param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NTCHW)
        {
            // todo for 3D model
            return run_3d_cuda(frame_rois, blob, param);
        }
        else
        {
            CLOG(ERROR, PRELOG) << "output_format not supported now";
            assert(0);
        }
    }
    else
    {
        CLOG(ERROR, PRELOG) << "lib_type not supported now";
        assert(0);
    }
    return ZJV_STATUS_OK;
}


} // namespace ZJVIDEO {
