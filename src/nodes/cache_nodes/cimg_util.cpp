#include <vector>
#include "CImg/CImg.h"
#include "common/StatusCode.h"
#include "nodes/infer_nodes/InferDefine.h"
#include <algorithm>

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
            float scale = ZJ_MIN( (float)width/img.width(), (float)height/ img.height());
            
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
            float scale = ZJ_MAX( (float)width/img.width(), (float)height/ img.height());
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
            std::cout << "resize_type not supported now";
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
                std::cout << "channel not match, input is " << img.spectrum() << " required is " << channel;
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
            else if(channel_format == ZJV_PREPROCESS_CHANNEL_FORMAT_RGB)
            {
                // do nothing
            }
            else
            {
                std::cout << "channel_format not supported now";
                assert(0);
            }
        }

        return ZJV_STATUS_OK;
    }


    static int Cimg_normalize(CImg<float> & img, std::vector<float> mean, std::vector<float> std)
    {
        for (int c = 0; c < img.spectrum(); ++c) {
            CImg<float> channel = img.get_channel(c);
            channel -= mean[c];
            channel /= std[c];
            img.draw_image(0, 0, 0, c, channel);
        }
        return ZJV_STATUS_OK;
    }


    
    static int cimg_preprocess(std::shared_ptr<const FrameData> frame_data, Rect roi, CImg<float> & img_norm, PreProcessParameter & param)
    {

        int interp = 0;
        if(param.interp_type == ZJV_PREPROCESS_INTERP_LINEAR) interp = 3;
        else if(param.interp_type == ZJV_PREPROCESS_INTERP_NEAREST) interp = 1;
        else if(param.interp_type == ZJV_PREPROCESS_INTERP_CUBIC) interp = 5;
        else
        {
            interp = 3;
            std::cout<< __FILE__ << __LINE__ << "interp_type not supported now" << param.interp_type << "using default ZJV_PREPROCESS_INTERP_LINEAR"; 
        }

        // 2. 转为CIimg格式，格式为RRRRRRRRRRGGGGGGGGGGBBBBBBBBBBBBB，unsigned char
        const unsigned char* data = (unsigned char*)frame_data->data->cpu_data();

        CImg<unsigned char> img ;
        if(frame_data->format == ZJV_IMAGEFORMAT_RGB24)
        {   
            int c = frame_data->channel();
            img = CImg<unsigned char>(c, frame_data->width, frame_data->height, 1);
            assert(frame_data->data->size() == img.size());
            memcpy(img.data(), data, img.size());
            img.permute_axes("yzcx");
        }
        else if(frame_data->format == ZJV_IMAGEFORMAT_RGBP)
        {
            int c =  frame_data->channel();
            img = CImg<unsigned char>(frame_data->width, frame_data->height, 1, c);
            assert(frame_data->data->size() == img.size());
            memcpy(img.data(), data, img.size());
        }
        else if(frame_data->format == ZJV_IMAGEFORMAT_GRAY8)
        {
            int c = frame_data->channel();
            img = CImg<unsigned char>(frame_data->width, frame_data->height, 1, c);
            assert(frame_data->data->size() == img.size());
            memcpy(img.data(), data, img.size());
        }
        else
        {
            std::cout<< __FILE__ << __LINE__ <<  "frame_data format not supported now, only support RGB24 and PRGB24";
            assert(0);
        }
        

        // 3. 裁剪
        CImg<unsigned char> roi_img = img.get_crop(roi.x, roi.y, roi.x+roi.width - 1, roi.y+roi.height - 1);


        // 4. 缩放

        Cimg_resize(roi_img, param.resize_width, param.resize_height, interp, param.resize_type, param.letterbox_value);
        Cimg_cvtColor(roi_img, param.resize_channel, param.channel_format);
        // 5. 归一化
        CImg<float> img_float = roi_img;

        if(param.do_normalize)
        {
            Cimg_normalize(img_float, param.mean_value, param.std_value);
        }

        #if 0

        // 存为png图
        char name[128] = {0};
        snprintf(name, sizeof(name), "../data/roi_img_%05d.bmp", frame_data->frame_id);
        roi_img.save(name);

        CImgDisplay disp(roi_img,"My Image");
        while (!disp.is_closed()) {
            disp.wait();
            if (disp.is_key()) {
                std::cout << "Key pressed: " << disp.key() << std::endl;
            }
        }
        #endif

        if(param.output_format == ZJV_PREPROCESS_OUTPUT_FORMAT_NHWC)
        {
            img_float.permute_axes("cxyz");
        }

        img_norm = img_float;
        return ZJV_STATUS_OK;
    }


    static int get_scale_pad(Rect roi, PreProcessParameter & param,  float & scalex, float & scaley, float & padx, float & pady)
    {
        if(param.resize_type == ZJV_PREPROCESS_RESIZE_STRETCH)
        {
            scalex = (float)param.resize_width/roi.width;
            scaley = (float)param.resize_height/roi.height;
            padx = 0;
            pady = 0;
        }
        else if(param.resize_type == ZJV_PREPROCESS_RESIZE_LETTERBOX)
        {
            float scale = ZJ_MIN( (float)param.resize_width/roi.width, (float)param.resize_height/roi.height);
            scalex = scale;
            scaley = scale;
            padx = (param.resize_width - roi.width * scale) / 2;
            pady = (param.resize_height - roi.height * scale) / 2;
        }
        else if(param.resize_type == ZJV_PREPROCESS_RESIZE_FILL)
        {
            float scale = ZJ_MAX( (float)param.resize_width/roi.width, (float)param.resize_height/roi.height);
            scalex = scale;
            scaley = scale;
            padx = (param.resize_width - roi.width * scale) / 2;
            pady = (param.resize_height - roi.height * scale) / 2;
        }
        else
        {
            std::cout<< __FILE__ << __LINE__ << "resize_type not supported now";
            assert(0);
        }

        return ZJV_STATUS_OK;
    }

}