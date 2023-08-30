"""
输入两个 .npy或者.csv 文件
输出对比结果
"""

import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\1FromDesktop\Contests\Causal\code\trustworthyAI\gcastle")
from castle.metrics import MetricsDAG
from castle.common.plot_dag import GraphDAG

def load_csv_as_nparray(filename):
    data = pd.read_csv(filename, header=None).values
    return data

def main():
    if len(sys.argv) != 3:
        print("Usage: python ./compare.py true_graph.npy/.csv est_graph.npy/.csv")
        return

    true_graph_file = sys.argv[1]
    est_graph_file = sys.argv[2]

    if true_graph_file.endswith(".csv"):
        true_graph = load_csv_as_nparray(true_graph_file)
    else:
        true_graph = np.load(true_graph_file)
    print("true_graph:\n", true_graph)

    if est_graph_file.endswith(".csv"):
        est_graph = load_csv_as_nparray(est_graph_file)
    else:
        est_graph = np.load(est_graph_file)
    print("est_graph:\n", est_graph)

    print(f"Shape of true graph: {true_graph.shape}")
    print(f"est graph:\n{est_graph.shape}")

    _metrics = MetricsDAG(est_graph, true_graph)

    print("Performance:")
    print(_metrics.metrics)

    GraphDAG(est_graph, true_graph)

if __name__ == "__main__":
    main()
# 先git clone一下官方的库
# 命令应该是 git clone https://github.com/huawei-noah/trustworthyAI.git
# conda activate CML  # 环境，可以是自己的
# python ./compare.py true_graph.csv est_graph.npy  # 输入npy或者csv文件（0-1矩阵）
# 上面代码第4行的路径改成自己的就行