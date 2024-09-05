
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
#define FPS 200
#define STEP 4

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
        cnt++;
        if(cnt%STEP != 0) continue;

        std::this_thread::sleep_for(std::chrono::milliseconds(1000*STEP/FPS));
        std::shared_ptr<ZJVIDEO::FrameData> frame = std::make_shared<ZJVIDEO::FrameData>(
            FRAME_WIDTH, FRAME_HEIGHT, ZJVIDEO::ZJV_IMAGEFORMAT_GRAY8);

        frame->fps = FPS/STEP;
        frame->camera_id = camera_id;
        frame->frame_id = cnt;
        
        std::memcpy(frame->data->mutable_cpu_data(), buffer.data(), frame->data->size());
        
        func(frame);
    }    
    return 0;
}


int main()
{  

    signal(SIGINT, signalHandler);  
    std::cout<< "laser welding!\n" ;


    std::string cfg_file = "../configure/pipeline_welding.json";

    
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

    std::shared_ptr<ZJVIDEO::SetRunModeControlData> mode_control = std::make_shared<ZJVIDEO::SetRunModeControlData>();
    mode_control->set_mode(ZJVIDEO::ZJV_PIPELINE_RUN_MODE_LIVING);
    std::shared_ptr<ZJVIDEO::ControlData> base_mode = std::dynamic_pointer_cast<ZJVIDEO::ControlData>(mode_control);
    pipeline.control(base_mode);

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
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::vector<std::shared_ptr<ZJVIDEO::EventData> > datas;
        datas.clear();
        pipeline.get_output_data(datas);

        if(datas.size() == 0)
        {
            continue;
        }

        for(const auto & data : datas)
        {
            frame_id = data->frame->frame_id;

            for(const auto & extra : data->extras)
            {
                if(extra->data_name == "WeldResult")
                {
                    std::shared_ptr<const ZJVIDEO::WeldResultData> weld = std::dynamic_pointer_cast<const ZJVIDEO::WeldResultData>(extra);
                    if(weld->is_enable)
                    {
                        printf("WeldResult:     frame_id: %d, camera_id: %d, weld_status: %d, status_score: %f, weld_depth: %f, front_quality: %f, back_quality: %f\n", 
                            weld->frame_id, weld->camera_id, weld->weld_status, weld->status_score, weld->weld_depth, weld->front_quality, weld->back_quality);
                    }                
                }
            }            
        }
        if(frame_id % 24 == 0)
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