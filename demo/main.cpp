
#include <iostream>
#include "pipeline/Pipeline.h"
#include <chrono>
#include <thread>
#include <csignal>

volatile std::sig_atomic_t flag = 0;
void signalHandler(int signum) {
    flag = 1;
}


int main()
{  
    signal(SIGINT, signalHandler);  
    std::cout<< "Hello, World!\n" ;

    std::string cfg_file = "../configure/pipeline_sample.json";

    ZJVIDEO::Pipeline pipeline(cfg_file);

    std::cout<< "pipeline.init()\n" ;
    pipeline.init();
    std::cout<< "pipeline.start()\n" ;
    pipeline.start();
    

    std::vector<std::string> src_node_name = pipeline.get_src_node_name();
    std::vector<std::string> dst_node_name = pipeline.get_dst_node_name();
    
    // 打印源节点数量
    std::cout<< "src_node_name.size(): " << src_node_name.size()  <<std::endl;

    for (auto & name : src_node_name)
    {
        std::cout<< "src_node_name: " << name <<std::endl;
    }

    int frame_id = 0;
    while(!flag)
    {
        // std::cout<< frame_id<<std::endl;
        frame_id++;
        int camera_id = 0;
        for (auto & name : src_node_name)
        {
            std::shared_ptr<ZJVIDEO::FrameData> data =  std::make_shared<ZJVIDEO::FrameData>();
            data->frame_id = frame_id;
            data->camera_id = camera_id;
            std::shared_ptr<ZJVIDEO::FlowData> flowdata= std::make_shared<ZJVIDEO::FlowData>(data);
            pipeline.set_input_data(name, flowdata);
            camera_id++;
        }
        // 延时10ms
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        //pipeline.show_debug_info();
        for(auto & name :dst_node_name)
        {
            while (1)
            {
                std::shared_ptr<ZJVIDEO::FlowData> flowdata = nullptr;
                pipeline.get_output_data(name, flowdata);
                if (flowdata)
                {
                    std::cout<< "get_output_data: "<<flowdata->frame->camera_id<< " " << flowdata->frame->frame_id <<std::endl;
                }
                else
                {
                    break;
                }
            }
        }
    }

    pipeline.stop();


    return 0;
}