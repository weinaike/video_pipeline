
#include <iostream>
#include "pipeline/Pipeline.h"
#include <chrono>
#include <thread>
#include <csignal>
#include "CImg/CImg.h"

#include "opencv2/videoio.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip> // for std::setw and std::setfill
#define use_opencv 

volatile std::sig_atomic_t flag = 0;
void signalHandler(int signum) {
    flag = 1;
}


#define FRAME_WIDTH 640
#define FRAME_HEIGHT 512
#define CHANNELS 1

std::string video_path = "../data/video/107#.raw";



int input_worker(std::function<int(const std::shared_ptr<ZJVIDEO::FrameData> & )> func, int camera_id)
{   
    std::ifstream raw_video(video_path, std::ios::binary);
    if (!raw_video.is_open()) {
        std::cerr << "Error opening raw video file" << std::endl;
        return -1;
    }

    int frame_size = FRAME_WIDTH * FRAME_HEIGHT * CHANNELS;
    std::vector<char> buffer(frame_size);
    int cnt = 0;

    while (!flag) {
        if (!raw_video.read(buffer.data(), frame_size)) {
            raw_video.clear();
            raw_video.seekg(0, std::ios::beg);
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        std::shared_ptr<ZJVIDEO::FrameData> frame = std::make_shared<ZJVIDEO::FrameData>(
            FRAME_WIDTH, FRAME_HEIGHT, ZJVIDEO::ZJV_IMAGEFORMAT_GRAY8);

        frame->fps = 200;
        frame->camera_id = camera_id;
        frame->frame_id = cnt;
        
        std::memcpy(frame->data->mutable_cpu_data(), buffer.data(), frame->data->size());
        cnt++;
        func(frame);
    }    
    return 0;
}


int main()
{  
    signal(SIGINT, signalHandler);  
    std::cout<< "laser welding!\n" ;


    std::string cfg_file = "../configure/pipeline_welding_classification.json";

    
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


    std::shared_ptr<ZJVIDEO::SetLoggerLevelControlData> level = std::make_shared<ZJVIDEO::SetLoggerLevelControlData>();
    level->set_level(ZJVIDEO::ZJV_LOGGER_LEVEL_INFO);
    std::shared_ptr<ZJVIDEO::ControlData> base_level = std::dynamic_pointer_cast<ZJVIDEO::ControlData>(level);
    pipeline.control(base_level);


    std::vector<std::thread > threads;
    
    for(int i = 0; i < 1; i++)
    {

        std::thread t1(input_worker, 
               [&pipeline](const std::shared_ptr<ZJVIDEO::FrameData>& data) {
                   return pipeline.set_input_data(data);
               }, 
               i);

        // std::thread t1(input_worker, std::bind(&ZJVIDEO::Pipeline::set_input_data, &pipeline, std::placeholders::_1), i);
        threads.emplace_back(std::move(t1));
    }

    // pipeline.set_input_data(std::make_shared<ZJVIDEO::VideoData>(video_path, 0));

    int frame_id = 0;
    
    while(!flag)
    {
        std::vector< std::shared_ptr<const ZJVIDEO::BaseData> >datas;
        datas.clear();
        pipeline.get_output_data(datas);

        if(datas.size() == 0)
        {
            continue;
        }

        cil::CImg<unsigned char> img;
        for(const auto & data :datas)
        {
            if(data->data_name == "Frame")
            {                
                std::shared_ptr<const ZJVIDEO::FrameData> frame = std::dynamic_pointer_cast<const ZJVIDEO::FrameData>(data);
                if(frame->format == ZJVIDEO::ZJV_IMAGEFORMAT_GRAY8)
                {
                    img = cil::CImg<unsigned char>(frame->width, frame->height, 1, frame->channel());
                    memcpy(img.data(), frame->data->cpu_data(), img.size());
                }
                frame_id = frame->frame_id;
            }
        }


        for(const auto & data :datas)
        {
            if(data->data_name == "ClassifyResult")
            {
                std::shared_ptr<const ZJVIDEO::ClassifyResultData> result = std::dynamic_pointer_cast<const ZJVIDEO::ClassifyResultData>(data);

                for(int i = 0; i < result->detect_box_categories.size(); i++)
                {
                    int label = result->detect_box_categories[i].label;
                    float score = result->detect_box_categories[i].score;
                    printf("frame_id: %d, cls: %d, score: %f\n", frame_id, label, score);
                }
            }
        }

        if(frame_id % 25 == 0)
        {
            pipeline.show_debug_info();
        }
    }

    for(auto & t : threads)
    {
        if(t.joinable())   t.join();
    }

    pipeline.stop();


    return 0;
}