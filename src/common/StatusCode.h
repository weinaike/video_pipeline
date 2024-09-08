


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
    ZJV_STATUS_QUEUE_EMPTY = -13,  // 节点数据错误
    ZJV_STATUS_QUEUE_NOT_ENOUGH = -14,  // 节点数据错误
    ZJV_STATUS_NOT_IMPLEMENT = -15,  // 节点数据错误

    
};


#define ZJ_CHECK_RETURN(ret) \
    if ((ret) != 0) { \
        printf("error code[%d], in %s : %d\n", ret,  __FILE__, __LINE__); \
        return ret; \
    }

#define ZJ_CHECK_ASSERT(ret) \
    if ((ret) != 0) { \
        printf("error code[%d], in %s : %d\n", ret,  __FILE__, __LINE__); \
        assert(0); \
    }

}
#endif  // ZJVIDEO_STATUSCODE_H
