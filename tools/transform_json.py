import json
import os
## 导入export.json文件
with open('export.json', 'r') as f:
    data = json.load(f)


## 读取json文件中的数据
for key in data['drawflow']:
    configure = {}
    pipeline = {}
    pipeline['task_name'] = key
    pipeline['expand_pipe'] = True
    pipeline['channel_num'] = 1
    pipeline['nodes'] = []

    nodes = data['drawflow'][key]['data']

    # Print node information
    for node_id, node_info in nodes.items():
        new_node = {}
        new_node['node_type'] = node_info['class']
        new_node['node_name'] = 'Node' + str(node_id)
        new_node['channels'] = node_info['typenode']
        # 判断data字典是否为空
        if (node_info['data'] != {}):
            cfg_name = new_node['node_name'] + "_config.json"
            new_node['cfg_file'] = cfg_name
            configure[cfg_name] = node_info['data']
        else:
            new_node['cfg_file'] = ""

        new_node['input_datas'] = []
        new_node['output_datas'] = []
        for out_key, out_name in node_info['output_names'].items():
            meta = {}
            meta['meta_data_type'] = out_name
            new_node['output_datas'].append(meta)
        
        for in_key, in_connect in node_info['inputs'].items():
            
            for connect_key, connects in in_connect.items():
                for connect in connects:
                    meta = {}
                    meta['pre_node'] = "Node" + str(connect['node'])

                    ## 获取第id各节点的output_names
                    out_names  = nodes[connect['node']]['output_names']
                    meta['meta_data_type'] = out_names[connect['input']]
                    new_node['input_datas'].append(meta)


        pipeline['nodes'].append(new_node)

    # Write configure file
    dir = 'configures/'
    if(not os.path.exists(dir)):
        os.mkdir(dir)
    for cfg_name, cfg_info in configure.items():
        with open(dir + cfg_name, 'w') as f:
            json.dump(cfg_info, f, indent=4)
    
    # Write pipeline file
    with open(dir + key + '_pipeline.json', 'w') as f:
        json.dump(pipeline, f, indent=4)
