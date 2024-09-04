
#include <iostream>
#include "pipeline/Pipeline.h"
#include <chrono>
#include <thread>
#include <csignal>

#include "opencv2/videoio.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

volatile std::sig_atomic_t flag = 0;
void signalHandler(int signum) {
    flag = 1;
}

std::string pic_path = "../data/cat.bmp";
// std::string video_path = "../data/video/person_tracker.flv";
std::string video_path = "../data/video/0H5mnFcm2Kg.mp4.mkv";


std::string imagenet_file = "../data/synset.txt";

std::vector<std::pair<std::string, std::string>> load_synset(const std::string& filename) 
{
    std::vector<std::pair<std::string, std::string>> synset;
    std::ifstream file(filename);

    if (file.is_open()) {
        std::string line;

        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string id;
            std::string label;
            ss >> id; // 读取每行的第一个单词作为ID
            std::getline(ss, label); // 读取剩余的部分作为标签
            synset.push_back({id, label});
        }

        file.close();
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }

    return synset;
}

static const char* coco_labels[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
};

void HSVtoRGB(float h, float s, float v, float& r, float& g, float& b) {
    int i = h * 6;
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
}


int main()
{  
    std::vector<std::pair<std::string, std::string>> synset = load_synset(imagenet_file);

    signal(SIGINT, signalHandler);  
    std::cout<< "Hello, World!\n" ;

    // std::string cfg_file = "../configure/pipeline_sample_segment.json";
    // std::string cfg_file = "../configure/pipeline_sample_infer.json";
    // std::string cfg_file = "../configure/pipeline_sample.json";
    std::string cfg_file = "../configure/pipeline_sample_video.json";
    // std::string cfg_file = "../configure/pipeline_sample_3D_classification.json";
    ZJVIDEO::Pipeline pipeline(cfg_file);

    std::cout<< "pipeline.init()\n" ;
    pipeline.init();
    std::cout<< "pipeline.start()\n" ;
    pipeline.start();
    



    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // std::vector<std::string> src_node_name = pipeline.get_src_node_name();
    // std::vector<std::string> dst_node_name = pipeline.get_dst_node_name();
    
    // 打印源节点数量
    // std::cout<< "src_node_name.size(): " << src_node_name.size()  <<std::endl;

    pipeline.set_input_data(std::make_shared<ZJVIDEO::VideoData>(video_path, 0));

    std::shared_ptr<ZJVIDEO::SetLoggerLevelControlData> level = std::make_shared<ZJVIDEO::SetLoggerLevelControlData>();
    level->set_level(ZJVIDEO::ZJV_LOGGER_LEVEL_DEBUG);
    std::shared_ptr<ZJVIDEO::ControlData> base_level = std::dynamic_pointer_cast<ZJVIDEO::ControlData>(level);
    pipeline.control(base_level);


    std::shared_ptr<ZJVIDEO::SetRunModeControlData> mode_control = std::make_shared<ZJVIDEO::SetRunModeControlData>();
    mode_control->set_mode(ZJVIDEO::ZJV_PIPELINE_RUN_MODE_RECORDED);
    std::shared_ptr<ZJVIDEO::ControlData> base_mode = std::dynamic_pointer_cast<ZJVIDEO::ControlData>(mode_control);
    pipeline.control(base_mode);


    int frame_id = 0;
    
    unsigned char red[] = { 255, 0, 0 };  
    unsigned char blue[] = { 0, 0, 255 };  
    unsigned char white[] = { 255, 255, 255 };  
    while(!flag)
    {
        std::vector< std::shared_ptr<const ZJVIDEO::BaseData> >datas;
        datas.clear();
        pipeline.get_output_data(datas);
        // std::cout<<"datas.size(): " << datas.size() << std::endl;
        if(datas.size() == 0)
        {
            continue;
        }

        cv::Mat cv_img;
        for(const auto & data :datas)
        {
            if(data->data_name == "Frame")
            {                
                std::shared_ptr<const ZJVIDEO::FrameData> frame = std::dynamic_pointer_cast<const ZJVIDEO::FrameData>(data);
                if(frame->format == ZJVIDEO::ZJV_IMAGEFORMAT_RGB24)
                {
                    cv_img = cv::Mat(frame->height, frame->width, CV_8UC3);
                    std::memcpy(cv_img.data, frame->data->cpu_data(), frame->data->size());
                }
                else if(frame->format == ZJVIDEO::ZJV_IMAGEFORMAT_RGBP)
                {
                    std::cout<< "ZJV_IMAGEFORMAT_RGBP" << std::endl;
                }
                else
                {
                    std::cout<< "other" << std::endl;
                }
        
                frame_id = frame->frame_id;
                // opencv 画 帧号信息
                cv::putText(cv_img, std::to_string(frame_id), cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            }
        }


        for(const auto & data :datas)
        {
            if(data->data_name == "DetectResult")
            {
                std::shared_ptr<const ZJVIDEO::DetectResultData> detect_result = std::dynamic_pointer_cast<const ZJVIDEO::DetectResultData>(data);
                for(int i = 0; i < detect_result->detect_boxes.size(); i++)
                {
                    // 画框detect_boxes 画到cv_img
                    cv::rectangle(cv_img, cv::Point(detect_result->detect_boxes[i].x1, detect_result->detect_boxes[i].y1),
                        cv::Point(detect_result->detect_boxes[i].x2, detect_result->detect_boxes[i].y2), cv::Scalar(0, 0, 255), 2);
                    cv::putText(cv_img, coco_labels[detect_result->detect_boxes[i].label], cv::Point(detect_result->detect_boxes[i].x1, detect_result->detect_boxes[i].y1), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

                    if(detect_result->data_type == ZJVIDEO::ZJV_DATATYPE_DETECTRESULT_TRACK)
                    {

                        // 画框track_boxes 画到cv_img
                        cv::rectangle(cv_img, cv::Point(detect_result->detect_boxes[i].track_boxes[0].x, detect_result->detect_boxes[i].track_boxes[0].y),
                        cv::Point(detect_result->detect_boxes[i].track_boxes[0].x + detect_result->detect_boxes[i].track_boxes[0].width, detect_result->detect_boxes[i].track_boxes[0].y + detect_result->detect_boxes[i].track_boxes[0].height), cv::Scalar(255, 0, 0), 2);
                        cv::putText(cv_img, std::to_string(detect_result->detect_boxes[i].track_id), cv::Point(detect_result->detect_boxes[i].track_boxes[0].x, detect_result->detect_boxes[i].track_boxes[0].y + detect_result->detect_boxes[i].track_boxes[0].height), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 0), 2);
                        for(int j = 0; j < detect_result->detect_boxes[i].track_boxes.size(); j++)
                        {
                            int centerx = detect_result->detect_boxes[i].track_boxes[j].x + detect_result->detect_boxes[i].track_boxes[j].width/2;
                            int centery = detect_result->detect_boxes[i].track_boxes[j].y + detect_result->detect_boxes[i].track_boxes[j].height/2;
                            cv::circle(cv_img, cv::Point(centerx, centery), 2, cv::Scalar(255, 0, 0), 2);
                        }
                    }
                }
            }
            else if(data->data_name == "ClassifyResult")
            {
                std::shared_ptr<const ZJVIDEO::ClassifyResultData> result = std::dynamic_pointer_cast<const ZJVIDEO::ClassifyResultData>(data);

                for(int i = 0; i < result->obj_attr_info.size(); i++)
                {
                    int label = result->obj_attr_info[i].label;
                    float score = result->obj_attr_info[i].score;
                    std::cout << "cls: " << label << "  " << score << std::endl;
                }                
                // img.draw_text(50, 50, synset[label].second, red, 0, 1);
            }
            else if(data->data_name == "SegmentResult")
            {
                // std::shared_ptr<const ZJVIDEO::SegmentResultData> result = std::dynamic_pointer_cast<const ZJVIDEO::SegmentResultData>(data);
                // cil::CImg<unsigned char> mask(result->mask->width, result->mask->height, 1, 1, 0);
                // memcpy(mask.data(), result->mask->data->cpu_data(), mask.size()*sizeof(unsigned char));
                // cil::CImg<unsigned char> color_mask = create_color_mask(mask, 20);
                // overlay_mask(img, color_mask);

            }
        }

        // cv::resize(cv_img, cv_img, cv::Size(640, 480));

        cv::cvtColor(cv_img, cv_img, cv::COLOR_RGB2BGR);
        cv::imshow("result", cv_img);
        cv::waitKey(1);
        // std::cout<<frame_id<<std::endl;
        if(frame_id%25 == 0)
        {
            pipeline.show_debug_info();
        }

    }

    pipeline.stop();


    return 0;
}