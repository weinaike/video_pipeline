
#include <iostream>
#include "pipeline/Pipeline.h"
#include <chrono>
#include <thread>


int main()
{  
    std::cout<< "Hello, World!\n" ;

    std::string cfg_file = "../configure/pipeline_sample.json";

    ZJVIDEO::Pipeline pipeline(cfg_file);

    std::cout<< "pipeline.init()\n" ;
    pipeline.init();
    std::cout<< "pipeline.start()\n" ;
    pipeline.start();
    

    std::vector<std::string> src_node_name = pipeline.get_src_node_name();
    
    // 打印源节点数量
    std::cout<< "src_node_name.size(): " << src_node_name.size()  <<std::endl;

    for (auto & name : src_node_name)
    {
        std::cout<< "src_node_name: " << name <<std::endl;
    }

    std::cout<< "Press 'q' to exit\n" ;

    int frame_id = 0;
    // while (getchar() != 'q')
    while(1)
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
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        pipeline.show_debug_info();
    }

    pipeline.stop();


    return 0;
}