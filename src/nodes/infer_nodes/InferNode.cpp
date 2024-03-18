

#include "InferNode.h"
#include "../nlohmann/json.hpp"
#include "../CImg/CImg.h"
using namespace cimg_library;

namespace ZJVIDEO {

#define INFER_LOG "InferNode"

InferNode::InferNode(const NodeParam & param) : BaseNode(param)
{
    el::Loggers::getLogger(INFER_LOG);
    
    parse_configure(param.m_cfg_file);
    init();

    m_max_batch_size = 8; // 根据模型配置设置

    CLOG(INFO, INFER_LOG) << "InferNode::InferNode";
}

InferNode::~InferNode()
{
    CLOG(INFO, INFER_LOG) << "InferNode::~InferNode";
}

int InferNode::parse_configure(std::string cfg_file)
{
    std::ifstream i(cfg_file);
    if(i.is_open() == false)
    {
        CLOG(ERROR, INFER_LOG) << "open cfg_file failed";
        return ZJV_STATUS_ERROR;
    }
    nlohmann::json j;
    i >> j;
    // 1. 解析出 EngineParameter
    if (j.contains("model") && j.contains("infer") ) 
    {
        try {
            m_engine_param.m_model_name = j["model"]["model_name"];
            
            // 字符串转换为枚举类型
            std::string device = j["model"]["device"];
            if(device == "CPU") m_engine_param.m_device = CPU;
            else if(device == "GPU") m_engine_param.m_device = GPU;
            else m_engine_param.m_device = CPU;

            m_engine_param.m_dynamic = j["model"]["dynamic_batch"];
            m_engine_param.m_encrypt = j["model"]["encrypt"];
            m_engine_param.m_engine_type = j["model"]["backend"];   // 这个最重要
            m_engine_param.m_model_path =  j["model"]["model_file"];
            m_engine_param.m_param_path = j["model"]["weight_file"];
            m_engine_param.m_numThread = j["model"]["num_thread"];
        }
        catch (nlohmann::json::exception& e) {
            CLOG(ERROR, INFER_LOG) << "parse model failed" << e.what();
        }
        try {
            m_engine_param.m_input_node_name = j["infer"]["input_names"].get<std::vector<std::string>>();
        }
        catch (nlohmann::json::exception& e) {
            CLOG(ERROR, INFER_LOG) << "'input_names' is not an array"  << e.what();
        }
        
        try {
            m_engine_param.m_output_node_name = j["infer"]["output_names"].get<std::vector<std::string>>();
        }
        catch (nlohmann::json::exception& e)  {
            CLOG(ERROR, INFER_LOG) << "'output_names' is not an array"<< e.what();
        }

        try {
            std::vector<std::vector<int>> input_dims = j["infer"]["input_dims"].get<std::vector<std::vector<int>>>();
            assert(m_engine_param.m_input_node_name.size() == input_dims.size());
            for (int i = 0; i < m_engine_param.m_input_node_name.size(); i++)
            {
                m_engine_param.m_input_nodes[m_engine_param.m_input_node_name[i]] = input_dims[i];

                if(input_dims[i].size() < 4 || input_dims[i].size() > 5)
                {
                    CLOG(ERROR, INFER_LOG) << "input_dims size is not supported now, only support 4,5 dims now.";
                    assert(0);
                }
            }
        }catch (nlohmann::json::exception& e) {
            CLOG(ERROR, INFER_LOG) << "'input_dims' is not an array"<< e.what();
        }
        

    }

    // 2. 解析出 preprocess



    // 3. 解析出 postprocess

    return ZJV_STATUS_OK;
}

int InferNode::init()
{
    // 1. create engine
    m_engine = EngineRegister::CreateEngine(m_engine_param);
    if (m_engine == nullptr)
    {
        CLOG(ERROR, INFER_LOG) << "Create Engine Failed";
        return ZJV_STATUS_ERROR;
    }


    return ZJV_STATUS_OK;
}

int InferNode::process_batch( const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                                std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch)
{
    for(int i = 0; i < in_metas_batch.size(); i++)
    {
        std::vector<std::shared_ptr<BaseData>> out_metas;
        out_metas_batch.push_back(out_metas);
    }
    std::vector<std::shared_ptr<FrameROI>> frame_rois;
    prepare(in_metas_batch, frame_rois);

    for(int i = 0; i < frame_rois.size(); i += m_max_batch_size)
    {
        std::vector<std::shared_ptr<FrameROI>> batch_frame_rois;

        for(int j = 0; j < m_max_batch_size; j++)
        {
            if((i+j) >= frame_rois.size()) break;
            batch_frame_rois.push_back(frame_rois[i+j]);
        }
        std::vector<FBlob> inputs;
        preprocess(batch_frame_rois, inputs);
        std::vector<FBlob> outputs;
        infer(inputs, outputs);
        postprocess(outputs, batch_frame_rois);
    }

    summary(frame_rois, out_metas_batch);
    return ZJV_STATUS_OK;
}

int InferNode::prepare( const std::vector<std::vector<std::shared_ptr<const BaseData> >> & in_metas_batch, 
                            std::vector<std::shared_ptr<FrameROI>>  &frame_rois)
{
    if(in_metas_batch.size() == 0) return ZJV_STATUS_ERROR;

    for(int i = 0; i < in_metas_batch.size(); i++)
    {
        
        if(in_metas_batch[i].size() == 1)
        {
            if(in_metas_batch[i][0]->data_name == "Frame")
            {
                std::shared_ptr<const FrameData> frame_data = std::dynamic_pointer_cast<const FrameData>(in_metas_batch[i][0]);
                std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
                frame_roi->frame = frame_data;
                frame_roi->input_vector_id = i;
                frame_roi->roi.x = 0;
                frame_roi->roi.y = 0;
                frame_roi->roi.width = frame_data->width ;
                frame_roi->roi.height = frame_data->height;
                frame_rois.push_back(frame_roi);
            }
            else
            {
                CLOG(ERROR, INFER_LOG) << "in_metas_batch without Frame";
                assert(0);
            }
        }
        else
        {   
            std::shared_ptr<const FrameData> frame_data;
            std::shared_ptr<FrameROI> frame_roi = std::make_shared<FrameROI>();
            std::vector<Rect>rois;
            for(int j = 0; j < in_metas_batch[i].size(); j++)
            {
                
                if(in_metas_batch[i][j]->data_name == "Frame")
                {
                    frame_data = std::dynamic_pointer_cast<const FrameData>(in_metas_batch[i][j]);                  
                }
                else if(in_metas_batch[i][j]->data_name == "DetectResult")
                {
                    std::shared_ptr<const DetectResultData> roi_data = std::dynamic_pointer_cast<const DetectResultData>(in_metas_batch[i][j]);
                    for(int k = 0; k < roi_data->detect_boxes.size(); k++)
                    {
                        Rect roi = {0};
                        roi.x = roi_data->detect_boxes[k].x1;
                        roi.y = roi_data->detect_boxes[k].y1;
                        roi.width = roi_data->detect_boxes[k].x2 - roi_data->detect_boxes[k].x1;
                        roi.height = roi_data->detect_boxes[k].y2 - roi_data->detect_boxes[k].y1;
                        rois.push_back(roi);
                    }
                }
                else
                {
                    CLOG(ERROR, INFER_LOG) << "in_metas_batch without Frame or ROI";
                    assert(0);
                }
            }
            for(int j = 0; j < rois.size(); j++) 
            {
                frame_roi->frame = frame_data;
                frame_roi->roi = rois[j];
                frame_roi->input_vector_id = i;
                frame_rois.push_back(frame_roi);
            }
        }
    }
    return ZJV_STATUS_OK;
}

int InferNode::preprocess(std::vector<std::shared_ptr<FrameROI>>  &frame_rois, std::vector<FBlob> & inputs)
{
    
    for(int j = 0; j < m_engine->m_input_nodes.size(); j++)
    {
        std::string input_name = m_engine->m_input_node_name[j];
        std::vector<int> input_dims = m_engine->m_input_nodes[input_name];
        int bs = frame_rois.size();
        int width = 0;
        int height = 0;
        int channel = 1;
        int times = 0;
        if(input_dims.size() == 4)
        {
            channel = input_dims[1];
            height = input_dims[2];
            width = input_dims[3];
        }
        else if(input_dims.size() == 5)
        {
            times = input_dims[1];
            channel = input_dims[2];
            height = input_dims[3];
            width = input_dims[4];
        }
        input_dims[0] = bs;




        FBlob input_blob(input_dims);
        float * input_data = input_blob.mutable_cpu_data();
        assert(input_data != nullptr);
        int count = channel * height * width;
        // std::cout<<input_blob.count()<<" "<<count<<" " << input_blob.data()->size()<<std::endl;
        // for(int i = 0; i < input_dims.size(); i++)
        // {
        //     std::cout<<input_dims[i]<<" ";
        // }
        // std::cout<<std::endl;
        for(int i = 0; i < frame_rois.size(); i++)
        {
            frame_rois[i]->input_width = width;
            frame_rois[i]->input_height = height;
            frame_rois[i]->scale_x = (float)width/frame_rois[i]->roi.width;
            frame_rois[i]->scale_y = (float)height/ frame_rois[i]->roi.height;
            frame_rois[i]->padx = 0;
            frame_rois[i]->pady = 0;

            // // letterbox 模式
            // float scale = std::min((float)width/frame_rois[i]->roi.width, (float)height/ frame_rois[i]->roi.height);
            // frame_rois[i]->scale_x = scale;
            // frame_rois[i]->scale_y = scale;
            // frame_rois[i]->padx = (width - frame_rois[i]->roi.width * scale) / 2;
            // frame_rois[i]->pady = (height - frame_rois[i]->roi.height * scale) / 2;

            
            std::shared_ptr<FrameROI> frame_roi = frame_rois[i];
            std::shared_ptr<const FrameData> frame_data = frame_roi->frame;
            Rect roi = frame_roi->roi;
            
            const unsigned char* data = (unsigned char*)frame_data->data->cpu_data();

            CImg<unsigned char> img(frame_data->width, frame_data->height, 1, frame_data->channel);
            assert(frame_data->data->size() == img.size());
            memcpy(img.data(), data, img.size());
            CImg<unsigned char> roi_img = img.get_crop(roi.x, roi.y, roi.x+roi.width, roi.y+roi.height);
            
            if(roi_img.spectrum() == channel)
            {
                roi_img.resize(width, height);
            }
            else if(roi_img.spectrum() == 1 && 3 == channel)
            {
                // 1 channel to 3 channel
                roi_img.resize(width, height, 1, 3);
            }
            else if(roi_img.spectrum() == 3 && 1 == channel)
            {
                roi_img.resize(width, height);
                // rgb 3 channel to  gray  channel
                roi_img.RGBtoHSI().channel(2);
            }
            else
            {
                CLOG(ERROR, INFER_LOG) << "channel not match, input is " << roi_img.spectrum() << " required is " << channel;
                assert(0);
            }
            CImg<float> img_float = roi_img;


            img_float /= 255;

            #if 0
            CImgDisplay disp(img_float,"My Image");
            while (!disp.is_closed()) {
                disp.wait();
                if (disp.is_key()) {
                    std::cout << "Key pressed: " << disp.key() << std::endl;
                }
            }
            #endif

            // std::cout<<i<<" "<< img_float.pixel_type() <<" " <<img_float._width<<" "<<img_float._height<<" "<<img_float._depth<<" "<<img_float._spectrum<<" "<<img_float.size() <<std::endl;
            std::memcpy(input_data + count * i, (float * )img_float.data(), img_float.size() * sizeof(float));
        }
        inputs.push_back(input_blob);
    }


    return ZJV_STATUS_OK;
}
int InferNode::infer(std::vector<FBlob> & inputs, std::vector<FBlob> & outputs)
{   
    std::vector<void*> ins;
    std::vector<std::vector<int>> input_shape;
    for(int i = 0; i < inputs.size(); i++)
    {
        ins.push_back(inputs[i].mutable_cpu_data());
        input_shape.push_back(inputs[i].shape());
    }    
    std::vector<std::vector<float>> outs;
    std::vector<std::vector<int>> outputs_shape;

    m_engine->forward(ins, input_shape, outs, outputs_shape); 

    for(int i = 0; i < outs.size(); i++)
    {
        FBlob output_blob(outputs_shape[i]);
        float * output_data = output_blob.mutable_cpu_data();
        std::memcpy(output_data, outs[i].data(), outs[i].size() * sizeof(float));
        outputs.push_back(output_blob);
    }

    return ZJV_STATUS_OK;

}


static float IoU(const DetectBox& a, const DetectBox& b) {
    float interArea = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)) *
                      std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    float unionArea = (a.x2 - a.x1) * (a.y2 - a.y1) +
                      (b.x2 - b.x1) * (b.y2 - b.y1) -
                      interArea;
    return interArea / unionArea;
}

static void NMS(std::vector<DetectBox>& boxes, float nms_thresh) {
    std::sort(boxes.begin(), boxes.end(), [](const DetectBox& a, const DetectBox& b) {
        return a.score > b.score;
    });

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].score == 0) continue;
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (IoU(boxes[i], boxes[j]) > nms_thresh) {
                boxes[j].score = 0;
            }
        }
    }

    boxes.erase(std::remove_if(boxes.begin(), boxes.end(), [](const DetectBox& a) {
        return a.score == 0;
    }), boxes.end());
}

int InferNode::postprocess(const std::vector<FBlob> & outputs, std::vector<std::shared_ptr<FrameROI>> & frame_roi_results)
{
    for(int i = 0; i < outputs.size(); i++)
    {
        if(m_engine_param.m_output_node_name[i] != "output")
        {
            CLOG(ERROR, INFER_LOG) << "output node name not supported now";
            assert(0);
        }
      

        const float * output_data = outputs[i].cpu_data();
        std::vector<int> output_shape = outputs[i].shape();
        int bs = output_shape[0];
        int num = output_shape[1];
        int dim = output_shape[2];
        int nc = dim - 5; // 类别数量


        for(int j = 0; j < bs; j++)
        {
            std::shared_ptr<DetectResultData> detect_result_data = std::make_shared<DetectResultData>();
            detect_result_data->data_name = "DetectResult";
            detect_result_data->detect_boxes.clear();

            for(int k = 0; k < num; k++)
            {
                float score = output_data[j*num*dim + k*dim + 4];

                
                int max_index = -1;

                float max_obj_conf = 0.0;
                for (int t = 0; t < nc; t++) {
                    auto obj_conf = output_data[j*num*dim + k*dim + 5 + t];;
                    if (obj_conf >= max_obj_conf) {
                        max_obj_conf = obj_conf;
                        max_index = t;
                    }
                }

                float conf = max_obj_conf * score;


                if(conf < 0.5) continue;


                float x1 = output_data[j*num*dim + k*dim] - output_data[j*num*dim + k*dim + 2]/2;
                float y1 = output_data[j*num*dim + k*dim + 1] - output_data[j*num*dim + k*dim + 3]/2;
                float x2 = x1 + output_data[j*num*dim + k*dim + 2];
                float y2 = y1 + output_data[j*num*dim + k*dim + 3];
                DetectBox detect_box;
                detect_box.x1 = x1;
                detect_box.y1 = y1;
                detect_box.x2 = x2;
                detect_box.y2 = y2;
                detect_box.score = conf;
                detect_box.label = max_index;
                detect_result_data->detect_boxes.push_back(detect_box);
            }
            float nms_thresh = 0.2;
            NMS(detect_result_data->detect_boxes,nms_thresh ) ;
            // 打印结果

            Rect roi = frame_roi_results[j]->roi;

            float scalex  = frame_roi_results[j]->scale_x;
            float scaley  = frame_roi_results[j]->scale_y;
            int padx = frame_roi_results[j]->padx;
            int pady = frame_roi_results[j]->pady;

            for(int k = 0; k < detect_result_data->detect_boxes.size(); k++)
            {
                detect_result_data->detect_boxes[k].x1 = detect_result_data->detect_boxes[k].x1 / scalex + roi.x - padx / scalex;
                detect_result_data->detect_boxes[k].y1 = detect_result_data->detect_boxes[k].y1 / scaley + roi.y - pady / scaley;
                detect_result_data->detect_boxes[k].x2 = detect_result_data->detect_boxes[k].x2 / scalex + roi.x - padx / scalex;
                detect_result_data->detect_boxes[k].y2 = detect_result_data->detect_boxes[k].y2 / scaley + roi.y - pady / scaley;
                // CLOG(INFO, INFER_LOG) << "detect_boxes: " << detect_result_data->detect_boxes[k].x1 << " " << detect_result_data->detect_boxes[k].y1 << " " << detect_result_data->detect_boxes[k].x2 << " " << detect_result_data->detect_boxes[k].y2 << " " << detect_result_data->detect_boxes[k].score << " " << detect_result_data->detect_boxes[k].label;
            }          

            frame_roi_results[j]->result.push_back(detect_result_data);
        }
    }



    return ZJV_STATUS_OK;
}

int InferNode::summary(const std::vector<std::shared_ptr<FrameROI>>  &frame_rois, 
                        std::vector<std::vector<std::shared_ptr<BaseData>>> & out_metas_batch)
{
    for(int i = 0; i < out_metas_batch.size(); i++)        
    {
        auto & out_metas = out_metas_batch[i];
        for(auto & output_name: m_nodeparam.m_output_datas)
        {
            std::shared_ptr<BaseData> data = DataRegister::CreateData(output_name);
            // 合并同一帧，相同目标类型的结果，如果有必要，进行nms
            for(auto & frame_roi_result:frame_rois)
            {
                if(frame_roi_result->input_vector_id != i) continue;

                for(auto & result: frame_roi_result->result)
                {
                    if(result->data_name == output_name)
                    {
                        data->append(result);
                    }
                }
            }
            
            if(data->data_name == "DetectResult")
            {
                float nms_thresh = 0.2;
                std::shared_ptr<DetectResultData> detect_result_data_all = std::dynamic_pointer_cast<DetectResultData>(data);
                NMS(detect_result_data_all->detect_boxes,nms_thresh);  
            }
          
            out_metas.push_back(data);
        }
    }
    return ZJV_STATUS_OK;
}


REGISTER_NODE_CLASS(Infer)

} // namespace ZJVIDEO
