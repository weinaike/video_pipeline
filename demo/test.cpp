#include <iostream>
#include "pipeline/Pipeline.h"
#include "common/CommonDefine.h"
#include "nodes/BaseNode.h"
#include "logger/easylogging++.h"


void test(const std::vector<std::shared_ptr<const ZJVIDEO::FrameData> > &fs)
{
    for (auto & frame : fs)
    {
        // frame->width = 1920;
        printf("frame->width: %d\n", frame->width);
        // frame->m_extras.push_back(std::make_shared<ZJVIDEO::BaseData>(ZJVIDEO::ZJV_DATATYPE_UNKNOWN));
    }
}

void test2(const std::vector<const ZJVIDEO::FrameData *> &fs)
{
    for (auto & frame : fs)
    {
        printf("frame->width: %d\n", frame->width);
        // frame->width = 1920;
    }
}


int main()
{  
    ZJVIDEO::FrameData data ;
    data.width =100; 
    std::shared_ptr< ZJVIDEO::FrameData> t = std::make_shared<ZJVIDEO::FrameData>();
    t->width = 10;
    std::shared_ptr<const ZJVIDEO::FrameData> frame_data =t;    
    // frame_data = std::make_shared<ZJVIDEO::FrameData>(data);
    
    std::vector<std::shared_ptr<const ZJVIDEO::FrameData>> fs;
    fs.push_back(frame_data);
    test(fs);

    ZJVIDEO::FrameData  frame_;
    ZJVIDEO::FrameData  frame2_;
    const ZJVIDEO::FrameData* frame = &frame2_;

    frame = &frame_;
    std::vector<const ZJVIDEO::FrameData*> fs2;
    fs2.push_back(frame);
    test2(fs2);

    return 0;
}