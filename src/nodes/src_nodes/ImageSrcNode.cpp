

#include "ImageSrcNode.h"
#define IMAGESRC_LOG "ImageSrc"
namespace ZJVIDEO {

ImageSrcNode::ImageSrcNode(const NodeParam & param) : BaseNode(param)
{
    el::Loggers::getLogger(IMAGESRC_LOG);
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::ImageSrcNode";
}

ImageSrcNode::~ImageSrcNode()
{
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::~ImageSrcNode";
}

int ImageSrcNode::parse_configure(std::string cfg_file)
{
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::parse_configure";
    return 0;
}

int ImageSrcNode::init()
{
    CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::init";
    return 0;
}

int ImageSrcNode::process_single(const std::vector<std::shared_ptr<const BaseData> > & in_metas, 
                                std::vector<std::shared_ptr<BaseData> > & out_metas)
{
    for(const auto & in :in_metas)
    {
        if (in->data_name == "Frame")
        {
            std::shared_ptr<const FrameData> frame_data = std::dynamic_pointer_cast<const FrameData>(in);
            //深度拷贝一份FrameData

            std::shared_ptr<FrameData> frame_data_copy = std::make_shared<FrameData>(*frame_data);

            out_metas.push_back(frame_data_copy);
        }
        
    }
    

    // CLOG(INFO, IMAGESRC_LOG) << "ImageSrcNode::process_single";
    return 0;
}

REGISTER_NODE_CLASS(ImageSrc)

} // namespace ZJVIDEO
