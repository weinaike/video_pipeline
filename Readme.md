# 视频流水线分析项目

系统架构

1. 多通道流水线
2. 数据流管理
3. 节点算子调度

节点构建

1. 源节点
2. 中间节点
   1. 推理节点
3. 输出节点

日志管理

1. 日志等级控制

流水线接口管理

1. 数据输入
2. 结果输出

类型管理

1. 目标类型
2. 属性类型

流水线配置

1. pipeline关键参数
2. 各节点关键参数

数据结构管理

1. FlowData，数据流转的单位
2. FrameData，XXXResultData，XXXCacheData，节点输出数据，
3. EventData：流水线结果输出
