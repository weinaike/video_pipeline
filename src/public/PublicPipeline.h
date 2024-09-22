
#ifndef PUBLIC_PIPELINE_H
#define PUBLIC_PIPELINE_H

#include <memory>
#include <string>
#include <vector>
#include "PublicData.h"


namespace ZJVIDEO {
    class FrameData;  // Forward declaration
    class ControlData;  // Forward declaration
    class EventData;  // Forward declaration

    class PUBLIC_API PublicPipeline {
    public:
        PublicPipeline(){};
        virtual ~PublicPipeline() = default;

        virtual int init() = 0;
        virtual int start() = 0;
        virtual int stop() = 0;
        virtual int set_input_data(const std::shared_ptr<FrameData> & data) = 0;
        virtual int control(std::shared_ptr<ControlData>& data ) = 0;
        virtual int get_output_data(std::vector<std::shared_ptr<EventData>> & data) = 0;
        virtual int show_debug_info() = 0;
        // Add other public methods here

        static std::shared_ptr<PublicPipeline> create(std::string configFile);
    };
}

#endif // PUBLIC_PIPELINE_H