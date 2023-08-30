# 导入必要的库

import pandas as pd
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph

# 定义一个函数，把GeneralGraph格式的量转换为邻接矩阵（没有权重）
def graph_to_adj_matrix(graph: GeneralGraph, root: str):
    # 获取图中的节点数
    n_nodes = len(graph.nodes)
    # 创建一个n_nodes x n_nodes大小的零矩阵
    adj_matrix = np.zeros((n_nodes, n_nodes))
    # 遍历图中的每条边
    for edge in graph.get_graph_edges():
        # 获取边的起点和终点
        start, end = int(edge.node1.get_name()), int(edge.node2.get_name())
        # 在邻接矩阵中将对应位置设为1
        adj_matrix[start, end] = 1
    # 将邻接矩阵转换为DataFrame并保存为csv文件
    pd.DataFrame(adj_matrix).to_csv(root, index=False, header=False)

# 定义一个函数，用于对alarm.csv进行预处理
def alarm_data_process(alarm_root="./data/original/original.csv", TIME_WIN_SIZE=300):
    alarms = pd.read_csv(alarm_root, index_col=0)
    alarms = alarms.sort_values(by='start_timestamp')
    alarms['win_id'] = alarms['start_timestamp'].map(lambda elem: int(elem / TIME_WIN_SIZE))

    samples = alarms.groupby(['alarm_id', 'win_id'])['start_timestamp'].count().unstack('alarm_id')
    samples = samples.dropna(how='all').fillna(0)
    samples = samples.sort_index(axis=1)

    # 重置索引，去掉 win_id 列
    samples = samples.reset_index(drop=True)
    # 将samples的列索引重命名为time
    samples.index.name = 'time'
    # 将samples的索引加一，从1开始
    samples.index += 1
    # 使用to_csv()方法，将samples保存为csv文件
    samples.to_csv('./data/processed/alarm_only.csv')
    param_data = pd.read_csv("data/processed/alarm_only.csv", index_col=0)
    # replace(' ', '_') 将空格替换为下划线
    param_data.columns = param_data.columns.str.replace(' ', '_')
    return param_data

# 定义一个函数，用于对true_graph.npy进行预处理
def true_graph_process(true_root='./data/original/true_graph.npy'):
    """ Step 1: 载入 真实因果图 的数据 """
    true_graph = np.load(true_root)
    # 展示true_graph
    # print(f"true graph:\n {true_graph}")
    # print(f"\nshape of true graph: {true_graph.shape}")
    """ Step 2: 保存 真实因果图 """
    # 保存true_graph
    np.savetxt('./data/processed/true_graph.csv', true_graph, delimiter=',', fmt='%.1f')
    # """ Step 3: 检测 真实因果图 """
    # # 打印 true_graph 中的所有元素
    # unique_elements = np.unique(true_graph)
    # print(unique_elements)   # [0. 1.]
    # # 检查 true_graph 是否是对称阵
    # is_symmetric = np.allclose(true_graph, true_graph.T)
    # print(is_symmetric)  # False
    true_graph = np.loadtxt('./data/processed/true_graph.csv', delimiter=',')
    return true_graph
