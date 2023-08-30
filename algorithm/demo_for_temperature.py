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

import pandas as pd


if __name__ == '__main__':
    """ Step 1: 设置参数 """
    print("\nStep 1: 设置参数\n")
    param_method = "CBNB_w"  # CBNB_w NBCB_w CBNB_e NBCB_e GCMVL CCM PCMCI PCGCE VarLiNGAM
    param_tau_max = 5  # param_tau_max 表示 tau 的最大值， 也就是最大的滞后阶数
    # 滞后阶数是指，如果 tau = 1， 那么就是说，如果 X 的值发生了变化，那么 Y 的值会在下一个时刻发生变化
    param_sig_level = 0.05  # param_sig_level 表示显著性水平
    # 显著性水平是指，若 p-value 小于显著性水平，就拒绝原假设：即，认为 X 和 Y 之间存在因果关系

    f1_adjacency_list = []  # adjacency 表示邻接矩阵
    f1_orientation_list = []  # orientation 表示有向图
    percentage_of_detection_skeleton = 0  # percentage_of_detection_skeleton 表示骨架图的检测率

    """ Step 2: 读取数据集 """
    print("\nStep 2: 读取数据集\n")
    param_data = pd.read_csv("../data/temperature/temperature.csv", index_col=0)
    # replace(' ', '_') 将空格替换为下划线
    param_data.columns = param_data.columns.str.replace(' ', '_')
    print("\nparam_data:\n\n", param_data)

    """ Step 3: 生成真实的因果图 """
    print("\nStep 3: 生成真实的因果图\n")
    list_nodes = []  # list_nodes 表示节点列表
    for col_i in param_data.columns:  # param_data.columns 表示数据集的列名
        list_nodes.append(GraphNode(col_i))  # GraphNode() 表示节点
    # causal_graph_true 表示真实的因果图
    causal_graph_true = GeneralGraph(list_nodes)
    # Endpoint.TAIL 表示箭头的起点， Endpoint.ARROW 表示箭头的终点
    causal_graph_true.add_edge(Edge(GraphNode(param_data.columns[1]),
                                    GraphNode(param_data.columns[0]),
                                    Endpoint.TAIL, Endpoint.ARROW))

    print("\n", type(causal_graph_true), "\n")
    print("\ncausal_graph_true:\n\n", causal_graph_true)

    """Step 4: 选择算法，生成预测的因果图 """
    print("\nStep 4: 选择算法，生成预测的因果图\n")
    # param_method 表示方法
    if param_method == "NBCB_w":
        nbcb = NBCBw(param_data, param_tau_max, param_sig_level, model="linear",  indtest="linear", cond_indtest="linear")
        nbcb.run()
        causal_graph_hat = nbcb.causal_graph
    elif param_method == "CBNB_w":
        cbnb = CBNBw(param_data, param_tau_max, param_sig_level, model="linear", indtest="linear", cond_indtest="linear")
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


    """Step 5: 评估预测的因果图"""
    print("\nStep 5: 评估预测的因果图\n")
    fa = f1_score(causal_graph_hat, causal_graph_true, ignore_orientation=True)
    fo = f1_score(causal_graph_hat, causal_graph_true, ignore_orientation=False)

    print("F1 adjacency with desired graph= " + str(fa))

    print("F1 orientation with desired graph = " + str(fo))

    f1_adjacency_list.append(fa)
    f1_orientation_list.append(fo)

    # B_est = causal_graph_hat
    # B_true = causal_graph_true
    # print("B_est:\n", B_est)
    # print("B_true:\n", B_true)
    # nt_metrics = MetricsDAG(B_est, B_true)
    # print(f"The performance for notears: \n{nt_metrics.metrics}")
#
#     if causal_graph_true.get_graph_edges() == causal_graph_hat.get_graph_edges():
#         percentage_of_detection_skeleton = percentage_of_detection_skeleton + 1
#     print("Percentage so far with true graph= " + str(percentage_of_detection_skeleton) + "/" + str(i + 1))
