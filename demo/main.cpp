
#include <iostream>
#include "pipeline/Pipeline.h"
#include "common/CommonDefine.h"
#include "nodes/BaseNode.h"

int main()
{
    std::cout << "Hello, World!" << std::endl;

    std::string cfg_file = "../configure/pipeline_sample.json";

    ZJVIDEO::Pipeline pipeline(cfg_file);

    std::cout << "pipeline.init()" << std::endl;
    pipeline.init();
    std::cout << "pipeline.start()" << std::endl;
    pipeline.start();
    
    
    std::cout << "Press 'q' to exit" << std::endl;

    std::vector<std::string> src_node_name = pipeline.get_src_node_name();
    
    // 打印源节点数量
    std::cout << "src_node_name.size(): " << src_node_name.size() << std::endl;

    for (auto & name : src_node_name)
    {
        std::cout << "src_node_name: " << name << std::endl;
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