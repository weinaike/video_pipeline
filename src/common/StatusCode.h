


#ifndef ZJVIDEO_STATUSCODE_H
#define ZJVIDEO_STATUSCODE_H

namespace ZJVIDEO {

enum StatusCode {
    ZJV_STATUS_OK = 0,  // 正常

    // 节点状态
    ZJV_STATUS_ERROR = -1,  // 错误
    ZJV_STATUS_INIT_ERROR = -2,  // 初始化错误
    ZJV_STATUS_START_ERROR = -3,  // 启动错误
    ZJV_STATUS_STOP_ERROR = -4,  // 停止错误
    ZJV_STATUS_NODE_ERROR = -5,  // 节点错误
    ZJV_STATUS_NODE_INIT_ERROR = -6,  // 节点初始化错误
    ZJV_STATUS_NODE_START_ERROR = -7,  // 节点启动错误
    ZJV_STATUS_NODE_STOP_ERROR = -8,  // 节点停止错误
    ZJV_STATUS_NODE_CONNECT_ERROR = -9,  // 节点连接错误
    ZJV_STATUS_NODE_DISCONNECT_ERROR = -10,  // 节点断开连接错误
    ZJV_STATUS_NODE_CONFIG_ERROR = -11,  // 节点配置错误
    ZJV_STATUS_NODE_CONTROL_ERROR = -12,  // 节点控制错误
    ZJV_STATUS_NODE_DATA_ERROR = -13,  // 节点数据错误
    ZJV_STATUS_NODE_CONFIG_TYPE_ERROR = -14,  // 节点配置类型错误
    ZJV_STATUS_NODE_CONTROL_TYPE_ERROR = -15,  // 节点控制类型错误
    ZJV_STATUS_NODE_DATA_TYPE_ERROR = -16,  // 节点数据类型错误
    ZJV_STATUS_NODE_DATA_NAME_ERROR = -17,  // 节点数据名称错误
    ZJV_STATUS_NODE_DATA_CREATE_TIME_ERROR = -18,  // 节点数据创建时间错误
    ZJV_STATUS_NODE_DATA_ERROR_ERROR = -19,  // 节点数据错误错误
    ZJV_STATUS_NODE_DATA_FRAME_ERROR = -20,  // 节点数据帧错误
    ZJV_STATUS_NODE_DATA_CONTROL_ERROR = -21,  // 节点数据控制错误
    ZJV_STATUS_NODE_DATA_CONFIG_ERROR = -22,  // 节点数据配置错误
    ZJV_STATUS_NODE_DATA_UNKNOWN_ERROR = -23,  // 节点数据未知错误
    ZJV_STATUS_NODE_DATA_MAX_ERROR = -24,  // 节点数据最大错误
    ZJV_STATUS_NODE_DATA_UNKNOWN_TYPE_ERROR = -25,  // 节点数据未知类型错误
    ZJV_STATUS_NODE_DATA_UNKNOWN_NAME_ERROR = -26,  // 节点数据未知名称错误
    ZJV_STATUS_NODE_DATA_UNKNOWN_CREATE_TIME_ERROR = -27,  // 节点数据未知创建时间错误
    ZJV_STATUS_NODE_DATA_UNKNOWN_ERROR_ERROR = -28,  // 节点数据未知错误错误
    
};

}

#endif  // ZJVIDEO_STATUSCODE_H
