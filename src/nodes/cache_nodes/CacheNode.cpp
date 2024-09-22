

#include "CacheNode.h"
#include "nlohmann/json.hpp"
#include "cimg_util.cpp"
#include "cuda_util.cpp"
#include "common/ExtraData.h"

#define CACHE_LOG "Cache"
namespace ZJVIDEO {

CacheNode::CacheNode(const NodeParam & param) : BaseNode(param)
{
    m_logger = el::Loggers::getLogger(CACHE_LOG);
    el::Configurations conf;
    conf.setToDefault();
    // Get the format for Info level
    std::string infoFormat = conf.get(el::Level::Info, el::ConfigurationType::Format)->value();
    // Set the format for Debug level to be the same as Info level
    conf.set(el::Level::Debug, el::ConfigurationType::Format, infoFormat);
    el::Loggers::reconfigureLogger(m_logger, conf);

    m_batch_process = false;
    m_max_batch_size = 1;
    m_count = 0;
    m_append_count = 0;
    parse_configure(param.m_cfg_file);
}

CacheNode::~CacheNode()
{
    CLOG(INFO, CACHE_LOG) << "CacheNode::~CacheNode";
}

int CacheNode::parse_configure(std::string cfg_file)
{

    CLOG(INFO, CACHE_LOG) << "CacheNode::parse_configure";
    std::ifstream i(cfg_file);
    if(i.is_open() == false)
    {
        CLOG(ERROR, CACHE_LOG) << "open cfg_file failed";
        m_os_type = 0;
        m_frame_num = 16;
        m_step = 0;
        m_fps = 1.0;
        m_device_id = -1;
        m_transform = false;
        m_output_type = ZJV_DATATYPE_IMAGECACHE;

        return ZJV_STATUS_ERROR;
    }
    nlohmann::json j;
    i >> j;
    std::string type = j["type"];
    if(type == "continue")
    {
        m_os_type = ZJV_OUTPUT_STREAM_TYPE_CONTINUE;
    }
    else if(type == "trigger")
    {
        m_os_type = ZJV_OUTPUT_STREAM_TYPE_TRIGGER;
    }
    else
    {
        m_os_type = ZJV_OUTPUT_STREAM_TYPE_CONTINUE;
    }

    m_frame_num = j["num"];
    m_step = j["step"];
    m_fps = j["fps"];
    m_device_id = j["device"];

    std::string ot_type = j["output_type"];
    if(ot_type == "ImageCache")
    {
        m_output_type = ZJV_DATATYPE_IMAGECACHE;
    }
    else
    {
        m_output_type = ZJV_DATATYPE_FEATURECACHE;
    }


    if (j.contains("param") ) 
    {
        m_transform = true;
        try {
            m_param.do_normalize = j["param"]["normalize"];

            m_param.letterbox_value = j["param"]["letterbox_color"].get<std::vector<int>>();
            m_param.output_dims = j["param"]["output_diims"].get<std::vector<int>>();
            m_param.mean_value = j["param"]["mean"].get<std::vector<float>>();
            m_param.std_value = j["param"]["std"].get<std::vector<float>>();

            std::string output_format = j["param"]["output_format"];
            if(output_format == "NCHW")
            {
                m_param.resize_channel = m_param.output_dims[1];
                m_param.resize_height = m_param.output_dims[2];
                m_param.resize_width = m_param.output_dims[3];
                m_param.output_format = ZJV_PREPROCESS_OUTPUT_FORMAT_NCHW;
            }                     
            else 
            {
                CLOG(ERROR, CACHE_LOG)<<"output_format not supported now in " ;
                assert(0);
            }


            std::string resize_type = j["param"]["resize_type"];
            if(resize_type == "Stretch") m_param.resize_type = ZJV_PREPROCESS_RESIZE_STRETCH;
            else if(resize_type == "LetterBox") m_param.resize_type = ZJV_PREPROCESS_RESIZE_LETTERBOX;
            else if(resize_type == "Fill") m_param.resize_type = ZJV_PREPROCESS_RESIZE_FILL;
            else m_param.resize_type = ZJV_PREPROCESS_RESIZE_UNKNOWN;

            std::string interp_type = j["param"]["interp_mode"];
            if(interp_type == "Linear") m_param.interp_type = ZJV_PREPROCESS_INTERP_LINEAR;
            else if(interp_type == "Nearest") m_param.interp_type = ZJV_PREPROCESS_INTERP_NEAREST;
            else if(interp_type == "Cubic") m_param.interp_type = ZJV_PREPROCESS_INTERP_CUBIC;
            else m_param.interp_type = ZJV_PREPROCESS_INTERP_UNKNOWN;
            
            std::vector<float> roi = j["param"]["roi"].get<std::vector<float>>();
            m_roi.x = roi[0];
            m_roi.y = roi[1];
            m_roi.width = roi[2];
            m_roi.height = roi[3];

            if(j["param"].contains("channel_format"))
            {
                std::string channel_format = j["param"]["channel_format"];
                if(channel_format == "RGB") m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_RGB;
                else if(channel_format == "BGR") m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_BGR;
                else m_param.channel_format = ZJV_PREPROCESS_CHANNEL_FORMAT_UNKNOWN;
            }

        }
        catch (nlohmann::json::exception& e) {
            CLOG(ERROR, CACHE_LOG) << "parse imagecache failed" << e.what();
        }    

    }
    else
    {
        m_transform = false;
    }




    // 打印配置参数
    CLOG(INFO, CACHE_LOG) << "----------------CacheNode config-----------------";
    CLOG(INFO, CACHE_LOG) << "type:    [" << m_os_type << "]";
    CLOG(INFO, CACHE_LOG) << "num:     [" << m_frame_num << "]";
    CLOG(INFO, CACHE_LOG) << "step:    [" << m_step << "]";
    CLOG(INFO, CACHE_LOG) << "fps:     [" << m_fps << "]";
    CLOG(INFO, CACHE_LOG) << "device:  [" << m_device_id << "]";


    if(m_transform)
    {
        
        // 打印预处理配置参数
        CLOG(INFO, CACHE_LOG) << "------- cache param config ------------";
        CLOG(INFO, CACHE_LOG) << "resize_width   [" << m_param.resize_width << "]";
        CLOG(INFO, CACHE_LOG) << "resize_height  [" << m_param.resize_height<<"]";
        CLOG(INFO, CACHE_LOG) << "resize_channel [" << m_param.resize_channel<< "]";
        if(m_param.resize_channel == 3)
        {
            CLOG(INFO, CACHE_LOG) << "mean:          [" << m_param.mean_value[0] << "," 
                << m_param.mean_value[1] << "," << m_param.mean_value[2]<<"]";
            CLOG(INFO, CACHE_LOG) << "std:           [" << m_param.std_value[0] << "," 
                << m_param.std_value[1] << "," << m_param.std_value[2]<<"]";
            CLOG(INFO, CACHE_LOG) << "letterbox:     [" << m_param.letterbox_value[0] << "," 
                << m_param.letterbox_value[1] << "," << m_param.letterbox_value[2]<<"]";
        }
        else if(m_param.resize_channel == 1)
        {
            CLOG(INFO, CACHE_LOG) << "mean:      [" << m_param.mean_value[0]<<"]";
            CLOG(INFO, CACHE_LOG) << "std:       [" << m_param.std_value[0]<<"]";
        }
        else
        {
            CLOG(ERROR, CACHE_LOG) << "resize_channel not supported now";
        }
        

        CLOG(INFO, CACHE_LOG) << "do_normalize:  [" << m_param.do_normalize<<"]";
        CLOG(INFO, CACHE_LOG) << "resize_type:   [" << m_param.resize_type<<"]";
        CLOG(INFO, CACHE_LOG) << "interp_type:   [" << m_param.interp_type<<"]";
        CLOG(INFO, CACHE_LOG) << "channel_format:[" << m_param.channel_format<<"]";
        CLOG(INFO, CACHE_LOG) << "output_format: [" << m_param.output_format<<"]";     
        CLOG(INFO, CACHE_LOG) << "------- cache param config ------------";
  
    }
    else
    {
        CLOG(INFO, CACHE_LOG) << "no transform";
    }


    CLOG(INFO, CACHE_LOG) << "----------------CacheNode config-----------------";

    return 0;
}

int CacheNode::init()
{
    CLOG(INFO, CACHE_LOG) << "CacheNode::init";
    return 0;
}


int CacheNode::transfer_data(std::shared_ptr<const FrameData> in_frame_data, std::shared_ptr<FrameData> & out_frame_data)
{
    if(m_transform == false)
    {
        out_frame_data = std::make_shared<FrameData>(*in_frame_data);
        return 0;
    }
    
    // 转换
    if(m_param.do_normalize)
    {
        int width = in_frame_data->width;
        int height = in_frame_data->height;
        Rect roi;
        roi.x = m_roi.x * width;
        roi.y = m_roi.y * height;
        roi.width = m_roi.width * width;
        roi.height = m_roi.height * height;


        if(m_param.resize_channel == 1)
        {
            out_frame_data = std::make_shared<FrameData>(m_param.resize_width, m_param.resize_height, ZJV_IMAGEFORMAT_GRAY_FLOAT);
        }
        else
        {
            out_frame_data = std::make_shared<FrameData>(m_param.resize_width, m_param.resize_height, ZJV_IMAGEFORMAT_RGBP_FLOAT);
        }     
        

        if(m_device_id < 0) // cpu
        {      

            CImg<float> img_float;
            cimg_preprocess(in_frame_data, roi, img_float, m_param);    
            std::memcpy(out_frame_data->data->mutable_cpu_data(), (float * )img_float.data(), img_float.size() * sizeof(float));
        }
        else // gpu
        {

        #ifdef Enable_CUDA
            // 1. 提取图片
            out_frame_data->data->set_device_id(m_device_id);
            float *out_data = (float * )out_frame_data->data->mutable_gpu_data();

            float scale_x , scale_y, padx, pady;
            get_scale_pad(roi, m_param, scale_x, scale_y, padx, pady);


            float matrix_2_3[2][3];
            matrix_2_3[0][0] = scale_x;
            matrix_2_3[0][1] = 0;
            matrix_2_3[0][2] = padx;
            matrix_2_3[1][0] = 0;
            matrix_2_3[1][1] = scale_y;
            matrix_2_3[1][2] = pady;

            cuda_preprocess(in_frame_data, roi, out_data, matrix_2_3, m_param, m_device_id);
        #else        
            CImg<float> img_float;
            cimg_preprocess(in_frame_data, roi, img_float, m_param);    
            std::memcpy(out_frame_data->data->mutable_cpu_data(), (float * )img_float.data(), img_float.size() * sizeof(float));
        #endif

        }


    }
    else
    {
        // 不归一化，暂时不做处理
        out_frame_data = std::make_shared<FrameData>(*in_frame_data);
    }  

    return 0;
}


int CacheNode::process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas)
{
    int output_interval_num = 1;
    std::shared_ptr<const FrameData> in_frame_data = nullptr;
    for(const auto & in :in_metas)
    {
        if (in->data_name == "Frame")
        {            
            in_frame_data = std::dynamic_pointer_cast<const FrameData>(in);
            output_interval_num = in_frame_data->fps/m_fps > output_interval_num ? in_frame_data->fps/m_fps : output_interval_num;
        }
        else
        {
            // trigger condition
        }
    }
    // CLOG(DEBUG, CACHE_LOG) << "output_interval_num: " << output_interval_num << "  frame_id : " << in_frame_data->frame_id
    //                         << " m_step: " << m_step << " m_count: " << m_count << " m_append_count: " << m_append_count 
    //                         << " m_frame_datas.size(): " << m_frame_datas.size();
    
    if(m_append_count % (m_step + 1) == 0)
    {    
        m_count++;
        // add to list
        std::shared_ptr<FrameData> frame_data = nullptr;
        transfer_data(in_frame_data, frame_data);
        // CLOG(DEBUG, CACHE_LOG) << "frame_data size: " <<  frame_data->data->size();
        m_frame_datas.push_back(frame_data);    
        m_append_count = 0;

        if(m_frame_datas.size() > 32)
        {
            m_frame_datas.pop_front();
        }
    }
    m_append_count++;

    // output

    if(m_os_type == ZJV_OUTPUT_STREAM_TYPE_CONTINUE)
    {
        if(m_output_type == ZJV_DATATYPE_IMAGECACHE)
        {
            std::shared_ptr<ImageCacheData> out = std::make_shared<ImageCacheData>();
            if(m_frame_datas.size() >= m_frame_num && m_count >= output_interval_num)
            {
                auto it = m_frame_datas.begin();
                for(int i  = 0 ; i < m_frame_num; i++)
                {
                    //std::cout << " " << (*it)->frame_id;
                    out->images.push_back(*it);
                    it++;
                }
                //std::cout << std::endl;
                m_count = 0;
                // CLOG(DEBUG, CACHE_LOG) << "output cache: " << out->images.size();
            }
            out_metas.push_back(out);
        }
        else if (m_output_type == ZJV_DATATYPE_FEATURECACHE)
        {
            std::shared_ptr<FeatureCacheData> feat = std::make_shared<FeatureCacheData>();
            if(m_frame_datas.size() >= m_frame_num && m_count >= output_interval_num)
            {
                std::vector<int> shape = m_param.output_dims;
                shape[0] = m_frame_num;
                std::shared_ptr<FBlob> out = std::make_shared<FBlob>(shape);
                float * out_data = (float*) out->mutable_cpu_data();               
                int sz = m_param.resize_channel * m_param.resize_height * m_param.resize_width;
                
                auto it = m_frame_datas.begin();
                assert(sz * 4 ==  (*it)->data->size());
                
                for(int i  = 0 ; i < m_frame_num; i++)
                {                    
                    memcpy(out_data + i * sz, (*it)->data->cpu_data(), sz * sizeof(float));
                    it++;
                }
                
                feat->feature = out;
                m_count = 0;
            }
            out_metas.push_back(feat);

        }
        else
        {
            CLOG(ERROR, CACHE_LOG) << "output_type not supported now";
            assert(0);
        }
    }
    else
    {            
        //todo trigger condition
        assert(0);            
    }
 

    // CLOG(INFO, CACHE_LOG) << "CacheNode::process_single";
    return 0;
}

REGISTER_NODE_CLASS(Cache)

} // namespace ZJVIDEO
