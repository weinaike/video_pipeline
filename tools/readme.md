# 流水线可视化配置工具


## 步骤：

1. 浏览器打开index.html, 配置流水线与参数
2. export 导出配置文件
3. 执行 transform_json.py 对导出配置格式转换为库所需格式
4. 将configures的加载到库中即可。（注意路径关系）


## 新增节点类型

若需新增节点类型， 在dist/node.js中添加相应节点内容， index.html中添加显示内容即可
