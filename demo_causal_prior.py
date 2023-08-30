from algorithm.cbnb_e import CBNBe
from algorithm.nbcb_e import NBCBe
from algorithm.nbcb_w import NBCBw
from algorithm.cbnb_w import CBNBw
from algorithm import baselines

from algorithm.evalusation_measure import f1_score
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

import numpy as np

from utils import graph_to_adj_matrix, alarm_data_process, true_graph_process

if __name__ == '__main__':
    """ Step 1: 设置参数 """
    print("\n\nStep 1: 设置参数")
    param_method = "VarLiNGAM"  # CBNB_w NBCB_w CBNB_e NBCB_e GCMVL CCM PCMCI PCGCE VarLiNGAM
    param_tau_max = 5  # param_tau_max 表示 tau 的最大值， 也就是最大的滞后阶数
    # 滞后阶数是指，如果 tau = 1， 那么就是说，如果 X 的值发生了变化，那么 Y 的值会在下一个时刻发生变化
    param_sig_level = 0.05  # param_sig_level 表示显著性水平
    # 显著性水平是指，若 p-value 小于显著性水平，就拒绝原假设：即，认为 X 和 Y 之间存在因果关系

    f1_adjacency_list = []  # adjacency 表示邻接矩阵
    f1_orientation_list = []  # orientation 表示有向图
    percentage_of_detection_skeleton = 0  # percentage_of_detection_skeleton 表示骨架图的检测率

    """ Step 2: 读取数据集 """
    print("\n\nStep 2: 读取数据集")
    param_data = alarm_data_process("./data/original/alarm.csv", 300)
    print("\nparam_data:\n", param_data)

    # 读取causal_prior.npy文件
    causal_prior = np.load("data/original/causal_prior.npy")
    selected_links = {}
    # 遍历causal_prior矩阵的每一行和每一列
    for i in range(causal_prior.shape[0]):
        for j in range(causal_prior.shape[1]):
            # 如果矩阵元素为1，表示存在因果关系
            # 如果矩阵元素为-1，表示可能存在因果关系
            if causal_prior[i, j] == 1 or causal_prior[i, j] == -1:
                # 如果字典中还没有键i，创建一个空列表
                if i not in selected_links:
                    selected_links[i] = []
                # 向列表中添加元组(j, 0)，表示同时链接
                selected_links[i].append((j, 0))
        if i not in selected_links:
            selected_links[i] = []
    # 打印结果
    print("\nselected_links:\n", selected_links)

    """ Step 3: 生成真实的因果图 """
    print("\n\nStep 3: 生成真实的因果图")

    # temperature 中的例子
    # list_nodes = []  # list_nodes 表示节点列表
    # for col_i in param_data.columns:  # param_data.columns 表示数据集的列名
    #     list_nodes.append(GraphNode(col_i))  # GraphNode() 表示节点
    # # causal_graph_true 表示真实的因果图
    # causal_graph_true = GeneralGraph(list_nodes)
    # # Endpoint.TAIL 表示箭头的起点， Endpoint.ARROW 表示箭头的终点
    # causal_graph_true.add_edge(Edge(GraphNode(param_data.columns[1]),
    #                                 GraphNode(param_data.columns[0]),
    #                                 Endpoint.TAIL, Endpoint.ARROW))
    # print("\ncausal_graph_true:\n\n", causal_graph_true)

    # 获取param_data的列索引，作为节点名称
    list_nodes = []  # list_nodes 表示节点列表
    for col_i in param_data.columns:  # param_data.columns 表示数据集的列名
        list_nodes.append(GraphNode(col_i))  # GraphNode() 表示节点
    graph = GeneralGraph(nodes=list_nodes)
    # 读入true_graph
    true_graph = true_graph_process('./data/original/true_graph.npy')
    # 获取矩阵的大小，即节点的个数
    n = true_graph.shape[0]
    # 遍历矩阵的每个元素，如果为1，就添加一条有向边到图中
    for i in range(n):
        for j in range(n):
            if true_graph[i, j] == 1:
                # 创建两个GraphNode对象，表示边的起点和终点
                node1 = list_nodes[i]
                node2 = list_nodes[j]
                # 创建一个Edge对象，表示有向边，用Endpoint.ARROW表示箭头
                edge = Edge(node1, node2, Endpoint.TAIL, Endpoint.ARROW)
                # print("\nedge:\n", edge)
                # 将边添加到图中
                graph.add_edge(edge)
    # 打印结果，查看图的类型和内容
    print("\ncausal_graph_true:\n\n", graph)
    causal_graph_true = graph

    """Step 4: 选择算法，生成预测的因果图 """
    print("\n\nStep 4: 选择算法，生成预测的因果图")
    # param_method 表示方法
    if param_method == "NBCB_w":
        nbcb = NBCBw(param_data, param_tau_max, param_sig_level, model="linear",  indtest="linear", cond_indtest="linear",
                     selected_links=selected_links)
        nbcb.run()
        causal_graph_hat = nbcb.causal_graph
    elif param_method == "CBNB_w":
        cbnb = CBNBw(param_data, param_tau_max, param_sig_level, model="linear", indtest="linear", cond_indtest="linear",
                     selected_links=selected_links)
        cbnb.run()
        causal_graph_hat = cbnb.causal_graph
    elif param_method == "NBCB_e":
        nbcb = NBCBe(param_data, param_tau_max, param_sig_level, model="linear",  indtest="linear", cond_indtest="linear")
        nbcb.run()
        causal_graph_hat = nbcb.causal_graph
    elif param_method == "CBNB_e":
        cbnb = CBNBe(param_data, param_tau_max, param_sig_level, model="linear", indtest="linear", cond_indtest="linear")
        cbnb.run()
        causal_graph_hat = cbnb.causal_graph
    elif param_method == "GCMVL":
        causal_graph_hat = baselines.granger_lasso(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
    elif param_method == "CCM":
        causal_graph_hat = baselines.ccm(param_data, tau_max=param_tau_max)
    elif param_method == "PCMCI":
        causal_graph_hat = baselines.pcmciplus(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
    elif param_method == "PCGCE":
        causal_graph_hat = baselines.pcgce(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
    elif param_method == "VarLiNGAM":
        causal_graph_hat = baselines.varlingam(param_data, tau_max=param_tau_max, sig_level=param_sig_level)
    elif param_method == "TiMINO":
        causal_graph_hat = baselines.run_timino_from_r([[param_data, "data"], [param_sig_level, "alpha"], [param_tau_max, "nlags"]])
    else:
        causal_graph_hat = None

    print(causal_graph_hat)  # causal_graph_hat 表示预测的因果图
    save_file = "./output/causal_prior_" \
                + str(param_method) + "_" \
                + str(param_tau_max) + "_" \
                + str(param_sig_level) \
                + ".csv"
    graph_to_adj_matrix(causal_graph_hat, save_file)


    """Step 5: 评估预测的因果图"""
    print("\n\nStep 5: 评估预测的因果图")
    fa = f1_score(causal_graph_hat, causal_graph_true, ignore_orientation=True)
    fo = f1_score(causal_graph_hat, causal_graph_true, ignore_orientation=False)

    print("F1 adjacency with desired graph= " + str(fa))

    print("F1 orientation with desired graph = " + str(fo))

    f1_adjacency_list.append(fa)
    f1_orientation_list.append(fo)

#
#     if causal_graph_true.get_graph_edges() == causal_graph_hat.get_graph_edges():
#         percentage_of_detection_skeleton = percentage_of_detection_skeleton + 1
#     print("Percentage so far with true graph= " + str(percentage_of_detection_skeleton) + "/" + str(i + 1))
