#include "PreProcessor.h"
#include "../../logger/easylogging++.h"

#include "../../CImg/CImg.h"

#define PRELOG "PreProc"
using namespace cimg_library;

namespace ZJVIDEO
{

    
static int Cimg_resize(CImg<unsigned char> & img, int width, int height, int interp_type, int resize_type, std::vector<int> letterbox_value)
{
    if(resize_type == ZJV_PREPROCESS_RESIZE_STRETCH)
    {
        img.resize(width, height, 1, img.spectrum(), interp_type);
    }
    else if(resize_type == ZJV_PREPROCESS_RESIZE_LETTERBOX)
    {
        float scale = std::min( (float)width/img.width(), (float)height/ img.height());
        
        int new_width = int(img.width() * scale)/2*2;
        int new_height = int(img.height() * scale)/2*2;
        img.resize(new_width, new_height, 1, img.spectrum(), interp_type);
        
        CImg<unsigned char> new_img(width, height, 1, img.spectrum(), letterbox_value[0]);
        int x = (width - new_width) / 2;
        int y = (height - new_height) / 2;
        new_img.draw_image(x, y, img);  // 将小图像绘制到大图像的指定位置
        img = new_img;
    }
    else if(resize_type == ZJV_PREPROCESS_RESIZE_FILL)
    {
        float scale = std::max( (float)width/img.width(), (float)height/ img.height());
        int new_width = int(img.width() * scale)/2*2;
        int new_height = int(img.height() * scale)/2*2;
        img.resize(new_width, new_height, 1, img.spectrum(), interp_type);

        int x1 = (new_width - width) / 2;
        int y1 = (new_height - height) / 2;
        // crop -1
        img.crop(x1, y1, x1 + width - 1, y1 + height - 1);
        // 输出img维度，宽高通道
    }
    else
    {
        CLOG(ERROR, PRELOG) << "resize_type not supported now";
        assert(0);
    }

    return ZJV_STATUS_OK;
}

static int Cimg_cvtColor(CImg<unsigned char> & img, int channel, int channel_format)
{
    if(img.spectrum() != channel) 
    {
        if(channel == 3 && img.spectrum() == 1)
        {
            img.resize(img.width(), img.height(), 1, 3);
        }
        else if(channel == 1 && img.spectrum() == 3)
        {
            img.RGBtoHSI().channel(2);
        }
        else
        {
            CLOG(ERROR, PRELOG) << "channel not match, input is " << img.spectrum() << " required is " << channel;
            assert(0);
        }
    }

    if(channel == 3 )
    {
        if(channel_format == ZJV_PREPROCESS_CHANNEL_FORMAT_BGR)
        {

            CImg<unsigned char> bgr_image(img.width(), img.height(), img.depth(), img.spectrum());

            bgr_image.draw_image(0, 0, 0, 0, img.get_channel(2));  // B
            bgr_image.draw_image(0, 0, 0, 1, img.get_channel(1));  // G
            bgr_image.draw_image(0, 0, 0, 2, img.get_channel(0));  // R

            img = bgr_image;
        }
    }
    else
    {
        CLOG(ERROR, PRELOG) << "channel_format is " << channel_format << "is not supported,  for channel is "<< channel;
    }


    return ZJV_STATUS_OK;
}


static int Cimg_normalize(CImg<float> & img, int dtype, std::vector<float> mean, std::vector<float> std)
{
    for (int c = 0; c < img.spectrum(); ++c) {
        CImg<float> channel = img.get_channel(c);
        channel -= mean[c];
        channel /= std[c];
        img.draw_image(0, 0, 0, c, channel);
    }
    return ZJV_STATUS_OK;
}


PreProcessor::PreProcessor(int lib_type = ZJV_PREPROCESS_LIB_CIMG): m_lib_type(lib_type)
{
    el::Loggers::getLogger(PRELOG);
}

int PreProcessor::parse_configure(const std::string & cfg_file)
{
    std::ifstream i(cfg_file);
    if(i.is_open() == false)
    {
        CLOG(ERROR, PRELOG) << "open cfg_file failed";
        return ZJV_STATUS_ERROR;
    }
    nlohmann::json j;
    i >> j;

    if (j.contains("preprocess") ) 
    {
        try {
            m_param.letterbox_value = j["preprocess"]["letterbox_color"].get<std::vector<int>>();
            m_param.do_normalize = j["preprocess"]["normalize"];
            m_param.resize_width = j["preprocess"]["resize_width"];
            m_param.resize_height = j["preprocess"]["resize_height"];
            m_param.resize_channel = j["preprocess"]["resize_channel"];
            m_param.mean_value = j["preprocess"]["mean"].get<std::vector<float>>();
            m_param.std_value = j["preprocess"]["std"].get<std::vector<float>>();
            if( m_param.mean_value.size() != m_param.resize_channel 
                || m_param.std_value.size() != m_param.resize_channel)
            {
                CLOG(ERROR, PRELOG) << "mean or std size not match resize_channel in " << cfg_file;
                assert(0);
            }

            std::string resize_type = j["preprocess"]["resize_type"];
            if(resize_type == "Stretch") m_param.resize_type = ZJV_PREPROCESS_RESIZE_STRETCH;
            else if(resize_type == "LetterBox") m_param.resize_type = ZJV_PREPROCESS_RESIZE_LETTERBOX;
            else if(resize_type == "Fill") m_param.resize_type = ZJV_PREPROCESS_RESIZE_FILL;
            else m_param.resize_type = ZJV_PREPROCESS_RESIZE_UNKNOWN;

            if(m_param.resize_type == ZJV_PREPROCESS_RESIZE_UNKNOWN ) 
            {
                CLOG(ERROR, PRELOG)<<"resize_type not supported now in " << cfg_file;
                assert(0);
            }

            std::string interp_type = j["preprocess"]["interp_mode"];
            if(interp_type == "Linear") m_param.interp_type = ZJV_PREPROCESS_INTERP_LINEAR;
            else if(interp_type == "Nearest") m_param.interp_type = ZJV_PREPROCESS_INTERP_NEAREST;
            else if(interp_type == "Cubic") m_param.interp_type = ZJV_PREPROCESS_INTERP_CUBIC;
            else m_param.interp_type = ZJV_PREPROCESS_INTERP_UNKNOWN;

            if(m_param.interp_type == ZJV_PREPROCESS_INTERP_UNKNOWN ) 
            {
                CLOG(ERROR, PRELOG)<< "interp_type not supported now in " << cfg_file;
                assert(0);
            }

            std::string channel_format = j["preprocess"]["channel_format"];
            if(channel_format == "RGB") m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_RGB;
            else if(channel_format == "BGR") m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_BGR;
            else m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_UNKNOWN;

            if(m_param.channel_format == ZJV_PREPROCESS_CHANNEL_FORMAT_UNKNOWN ) 
            {
                CLOG(ERROR, PRELOG)<<"channel_format not supported now in " << cfg_file;
                assert(0);
            }

            std::string dtype = j["preprocess"]["dtype"];
            if(dtype == "float32") m_param.dtype = ZJV_PREPROCESS_INPUT_DTYPE_FLOAT32;
            else if(dtype == "uint8") m_param.dtype = ZJV_PREPROCESS_INPUT_DTYPE_UINT8;
            else m_param.dtype = ZJV_PREPROCESS_INPUT_DTYPE_UNKNOWN;

            if(m_param.dtype == 0 ) 
            {
                CLOG(ERROR, PRELOG)<<"dtype not supported now in " << cfg_file;
                assert(0);
            }

            std::string input_format = j["preprocess"]["input_format"];
            if(input_format == "NCHW") m_param.input_format = ZJV_PREPROCESS_INPUT_FORMAT_NCHW;
            else if(input_format == "NHWC") m_param.input_format = ZJV_PREPROCESS_INPUT_FORMAT_NHWC;
            else m_param.input_format = ZJV_PREPROCESS_INPUT_FORMAT_UNKNOWN;

            if(m_param.input_format == ZJV_PREPROCESS_INPUT_FORMAT_UNKNOWN ) 
            {
                CLOG(ERROR, PRELOG)<<"input_format not supported now in " << cfg_file;
                assert(0);
            }

            // 打印预处理配置参数
            CLOG(INFO, PRELOG) << "------- preprocess config "<<cfg_file<<"------------";
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
            CLOG(INFO, PRELOG) << "dtype:         [" << m_param.dtype<<"]";
            CLOG(INFO, PRELOG) << "input_format:  [" << m_param.input_format<<"]";     
            CLOG(INFO, PRELOG) << "------- preprocess config "<<cfg_file<<"------------";

        }
        catch (nlohmann::json::exception& e) {
            CLOG(ERROR, PRELOG) << "parse preprocess failed" << e.what();
        }
    }
    
    return ZJV_STATUS_OK;
}


int PreProcessor::run_cimg(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param)
{

    int interp = 0;
    if(param.interp_type == ZJV_PREPROCESS_INTERP_LINEAR) interp = 3;
    else if(param.interp_type == ZJV_PREPROCESS_INTERP_NEAREST) interp = 1;
    else if(param.interp_type == ZJV_PREPROCESS_INTERP_CUBIC) interp = 5;
    else
    {
        interp = 3;
        CLOG(ERROR, PRELOG) << "interp_type not supported now" << param.interp_type << "using default ZJV_PREPROCESS_INTERP_LINEAR"; 
    }

    float * input_data = blob.mutable_cpu_data();
    assert(input_data != nullptr);
    int count = param.resize_channel * param.resize_height * param.resize_width;

    for(int i = 0; i < frame_rois.size(); i++)
    {
        if(param.resize_type == ZJV_PREPROCESS_RESIZE_STRETCH)
        {
            frame_rois[i]->input_width = param.resize_width;
            frame_rois[i]->input_height = param.resize_height;
            frame_rois[i]->scale_x = (float)param.resize_width/frame_rois[i]->roi.width;
            frame_rois[i]->scale_y = (float)param.resize_height/ frame_rois[i]->roi.height;
            frame_rois[i]->padx = 0;
            frame_rois[i]->pady = 0;
        }
        else if(param.resize_type == ZJV_PREPROCESS_RESIZE_LETTERBOX)
        {
            float scale = std::min( (float)param.resize_width/frame_rois[i]->roi.width, 
                                    (float)param.resize_height/ frame_rois[i]->roi.height);
            frame_rois[i]->scale_x = scale;
            frame_rois[i]->scale_y = scale;
            frame_rois[i]->padx = (param.resize_width - frame_rois[i]->roi.width * scale) / 2;
            frame_rois[i]->pady = (param.resize_height - frame_rois[i]->roi.height * scale) / 2;
        }
        else if(param.resize_type == ZJV_PREPROCESS_RESIZE_FILL)
        {
            float scale = std::max( (float)param.resize_width/frame_rois[i]->roi.width, 
                                    (float)param.resize_height/ frame_rois[i]->roi.height);
            frame_rois[i]->scale_x = scale;
            frame_rois[i]->scale_y = scale;
            frame_rois[i]->padx = (param.resize_width - frame_rois[i]->roi.width * scale) / 2;
            frame_rois[i]->pady = (param.resize_height - frame_rois[i]->roi.height * scale) / 2;
        }
        else
        {
            CLOG(ERROR, PRELOG) << "resize_type not supported now";
            assert(0);
        }
        
        // 1. 提取图片
        const std::shared_ptr<FrameROI> frame_roi = frame_rois[i];
        std::shared_ptr<const FrameData> frame_data = frame_roi->frame;
        Rect roi = frame_roi->roi;
        // 2. 转为CIimg格式，格式为RRRRRRRRRRGGGGGGGGGGBBBBBBBBBBBBB，unsigned char
        const unsigned char* data = (unsigned char*)frame_data->data->cpu_data();
        CImg<unsigned char> img(frame_data->width, frame_data->height, 1, frame_data->channel);            
        assert(frame_data->data->size() == img.size());
        memcpy(img.data(), data, img.size());
        // 3. 裁剪
        CImg<unsigned char> roi_img = img.get_crop(roi.x, roi.y, roi.x+roi.width - 1, roi.y+roi.height - 1);


        // 4. 缩放

        Cimg_resize(roi_img, param.resize_width, param.resize_height, interp, param.resize_type, param.letterbox_value);
        Cimg_cvtColor(roi_img, param.resize_channel, param.channel_format);
        // 5. 归一化
        CImg<float> img_float = roi_img;

        if(param.do_normalize)
        {
            Cimg_normalize(img_float, param.dtype, param.mean_value, param.std_value);
        }

        #if 0
        CImgDisplay disp(roi_img,"My Image");
        while (!disp.is_closed()) {
            disp.wait();
            if (disp.is_key()) {
                std::cout << "Key pressed: " << disp.key() << std::endl;
            }
        }
        #endif

        if(param.input_format == ZJV_PREPROCESS_INPUT_FORMAT_NHWC)
        {
            img_float.permute_axes("cxyz");
        }
        if(img_float.size() != count)
        {
            CLOG(ERROR, PRELOG) << "img_float size not match count" << img_float.size() << "!=" << count;
            assert(img_float.size() == count);
        }

        std::memcpy(input_data + count * i, (float * )img_float.data(), img_float.size() * sizeof(float));
    }


    return ZJV_STATUS_OK;

}

int PreProcessor::run_cuda(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param)
{
    return ZJV_STATUS_OK;
}

int PreProcessor::run(const std::vector<std::shared_ptr<FrameROI>> & frame_rois, FBlob & blob, PreProcessParameter & param)
{
    if(m_lib_type == ZJV_PREPROCESS_LIB_CIMG)
    {
        return run_cimg(frame_rois, blob, param);
    }
    else
    {
        CLOG(ERROR, PRELOG) << "lib_type not supported now";
        assert(0);
    }
}


} // namespace ZJVIDEO {
