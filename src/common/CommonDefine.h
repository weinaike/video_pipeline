


#ifndef ZJVIDEO_COMMMONDEFINE_H
#define ZJVIDEO_COMMMONDEFINE_H

#include "BaseData.h"
#include "ControlData.h"
#include "ConfigData.h"
#include "ThreadSaveQueue.h"
#include "StatusCode.h"
#include "AbstractNode.h"
#include "NodeParam.h"
#include "NodeFactory.h"
#include "FrameData.h"
#include "FlowMetaData.h"

// 通道重命名
static std::string join_string(std::string str, int i)
{
    return str + "_c" + std::to_string(i);
}
// 重命名后的通道解析, -1 表示不区分通道
static int parse_id(std::string str)
{
    std::size_t pos = str.rfind("_c");
    if (pos != std::string::npos) 
    {
        std::string i_str = str.substr(pos + 2); // 2 is the length of "_c"
        if (std::all_of(i_str.begin(), i_str.end(), ::isdigit)) {
            return std::stoi(i_str);
        } else {
            return -1;
        }
    } else {
        // Handle error: the string does not contain "_c"
        return -1;
    }
}


#endif  // ZJVIDEO_BASEDATA_H
