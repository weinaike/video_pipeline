#ifndef __ZJV_INFER_DEFINE_H__
#define __ZJV_INFER_DEFINE_H__

#include "common/CommonDefine.h"
#include "common/Blob.h"
#include "common/ExtraData.h"
#include "nlohmann/json.hpp"

namespace ZJVIDEO
{

    enum PreProcessResizeType
    {
        ZJV_PREPROCESS_RESIZE_UNKNOWN = 0,   // "Unknown"
        ZJV_PREPROCESS_RESIZE_STRETCH = 1,   // "Stretch"
        ZJV_PREPROCESS_RESIZE_LETTERBOX = 2, // "Letterbox"
        ZJV_PREPROCESS_RESIZE_FILL = 3,      // "Fill"
    };

    enum PreProcessInterpType
    {
        ZJV_PREPROCESS_INTERP_UNKNOWN = 0, // "Unknown"
        ZJV_PREPROCESS_INTERP_LINEAR = 1,  // "Linear"
        ZJV_PREPROCESS_INTERP_NEAREST = 2, // "Nearest"
        ZJV_PREPROCESS_INTERP_CUBIC = 3,   // "Cubic"
    };

    enum PreProcessOutputFormat
    {
        ZJV_PREPROCESS_OUTPUT_FORMAT_UNKNOWN    = 0,    // "Unknown"
        ZJV_PREPROCESS_OUTPUT_FORMAT_NCHW       = 1,    // "NCHW"
        ZJV_PREPROCESS_OUTPUT_FORMAT_NHWC       = 2,    // "NHWC"
        ZJV_PREPROCESS_OUTPUT_FORMAT_NCTHW      = 3,    // "NCTHW" for 3D model
        ZJV_PREPROCESS_OUTPUT_FORMAT_NTCHW      = 4,    // "NTCHW" for 3D model
        ZJV_PREPROCESS_OUTPUT_FORMAT_NTHW       = 5,    // "NTHW" for 3D model
    };

    enum PreProcessChannelFormat
    {
        ZJV_PREPROCESS_CHANNEL_FORMAT_UNKNOWN = 0, // "Unknown"
        ZJV_PREPROCESS_CHANNEL_FORMAT_RGB = 1,     // "RGB"
        ZJV_PREPROCESS_CHANNEL_FORMAT_BGR = 2,     // "BGR"
    };


    struct PreProcessParameter
    {
        std::string output_name;
        bool do_normalize;

        int resize_type; // PreProcessResizeType
        int interp_type; // PreProcessInterpType

        int channel_format; // PreProcessChannelFormat

        int output_format;   // PreProcessOutputFormat
        std::vector<float> mean_value;
        std::vector<float> std_value;
        std::vector<int> letterbox_value;
        std::vector<int> output_dims;

        int resize_width;
        int resize_height;
        int resize_channel;
    };

    enum PreProcessLibType
    {
        ZJV_PREPROCESS_LIB_UNKNOWN = 0, // "Unknown"
        ZJV_PREPROCESS_LIB_OPENCV = 1,  // "OpenCV"
        ZJV_PREPROCESS_LIB_CIMG = 2,    // "CImg"
        ZJV_PREPROCESS_LIB_CUDA = 3,    // "CUDA"
    };

    struct FrameROI
    {
        int input_vector_id;
        std::shared_ptr<const FrameData> frame; 
        std::vector<std::shared_ptr<const FrameData>> frames;   // 3D模型的多帧数据, 来源与ImageCache
        std::shared_ptr<const FBlob> fblob;                        // 3D模型的输入blob, 来源与ImageCache
        const DetectBox * original;                     // roi的原始目标
        // 原图坐标系下的roi
        Rect roi;
        // 网络输入宽
        int input_width;
        // 网络输入高
        int input_height;
        // 缩放比例x，roi宽/网络输入宽
        float scale_x;
        // 缩放比例y, roi高/网络输入高
        float scale_y;
        // 对于letterbox的缩放模式，填充起始点x，y
        int padx;
        int pady;
        int resize_type ;
        // 预处理指针
        PreProcessParameter * pre_process;
        // 模型推理结果，可以支持多种结果同时输出
        std::vector<std::shared_ptr<BaseData>> result;
    };

}

#endif // __ZJV_INFER_DEFINE_H__