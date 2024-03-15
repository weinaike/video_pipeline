
#include <iostream>
#include "pipeline/Pipeline.h"
#include <chrono>
#include <thread>
#include <csignal>
#include "CImg/CImg.h"

volatile std::sig_atomic_t flag = 0;
void signalHandler(int signum) {
    flag = 1;
}


int input_worker(std::function<int(const std::shared_ptr<ZJVIDEO::FrameData> & )> func, int camera_id)
{
    int cnt = 0;
    cimg_library::CImg<unsigned char> img("../data/test.bmp");
    // int camera_id = 0;
    while (!flag)
    {
        std::shared_ptr<ZJVIDEO::FrameData> frame= std::make_shared<ZJVIDEO::FrameData>();

        frame->width = img.width();
        frame->height = img.height();
        frame->channel = img.spectrum();
        frame->depth = 8;
        frame->format = ZJVIDEO::ZJV_IMAGEFORMAT_BGR24;
        frame->fps = 25;      
        frame->camera_id = camera_id;
        frame->frame_id = cnt;
        frame->data.reset(new ZJVIDEO::SyncedMemory(img.size(),img.data(), ZJVIDEO::ZJV_SYNCEHEAD_HEAD_AT_CPU));

        cnt++;
        func(frame);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    return 0;
}


int main()
{  


    auto time1 = std::chrono::system_clock::now();
    auto time2 = std::chrono::system_clock::now();

    if (time1 == time2) {
        std::cout << "The times are the same.\n";
    } else {
        std::cout << "The times are different.\n";
    }

    // return 0;


    signal(SIGINT, signalHandler);  
    std::cout<< "Hello, World!\n" ;

    std::string cfg_file = "../configure/pipeline_sample_infer.json";

    ZJVIDEO::Pipeline pipeline(cfg_file);

    std::cout<< "pipeline.init()\n" ;
    pipeline.init();
    std::cout<< "pipeline.start()\n" ;
    pipeline.start();
    

    // std::vector<std::string> src_node_name = pipeline.get_src_node_name();
    // std::vector<std::string> dst_node_name = pipeline.get_dst_node_name();
    
    // 打印源节点数量
    // std::cout<< "src_node_name.size(): " << src_node_name.size()  <<std::endl;


    std::vector<std::thread > threads;
    //
    for(int i = 0; i < 2; i++)
    {
        std::thread t1(input_worker, std::bind(&ZJVIDEO::Pipeline::set_input_data, &pipeline, std::placeholders::_1), i);
        threads.emplace_back(std::move(t1));
    }

    int frame_id = 0;
    while(!flag)
    {
        std::vector< std::shared_ptr<ZJVIDEO::EventData> >datas;
        datas.clear();
        pipeline.get_output_data(datas);
        for(const auto & data :datas)
        {
            std::cout<<  " camera_id: " << data->frame->camera_id << " frame_id: " << data->frame->frame_id << std::endl;
        
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // pipeline.show_debug_info();

    }

    for(auto & t : threads)
    {
        if(t.joinable())   t.join();
    }

    pipeline.stop();


    return 0;
}