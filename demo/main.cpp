
#include <iostream>
#include "pipeline/Pipeline.h"
#include "common/CommonDefine.h"
#include "nodes/BaseNode.h"
#include "logger/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

int main()
{  
    LOG(INFO) << "Hello, World!" ;

    std::string cfg_file = "../configure/pipeline_sample.json";

    ZJVIDEO::Pipeline pipeline(cfg_file);

    LOG(INFO) << "pipeline.init()" ;
    pipeline.init();
    LOG(INFO) << "pipeline.start()" ;
    pipeline.start();
    
    
    LOG(INFO) << "Press 'q' to exit" ;

    std::vector<std::string> src_node_name = pipeline.get_src_node_name();
    
    // 打印源节点数量
    LOG(INFO) << "src_node_name.size(): " << src_node_name.size() ;

    for (auto & name : src_node_name)
    {
        LOG(INFO) << "src_node_name: " << name ;
    }
    while (getchar() != 'q')
    {
        std::shared_ptr<ZJVIDEO::BaseData> data= std::make_shared<ZJVIDEO::BaseData>(ZJVIDEO::ZJV_DATATYPE_FRAME);

        for (auto & name : src_node_name)
        {
            pipeline.set_input_data(name, data);
        }
    }

    pipeline.stop();


    return 0;
}