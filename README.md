# CBNB_demo

论文：

Hybrids of Constraint-based and Noise-based Algorithms for Causal Discovery from Time Series

https://arxiv.org/pdf/2306.08765.pdf

开源代码仓库：

https://github.com/ckassaad/Hybrids_of_CB_and_NB_for_Time_Series

&nbsp;

仓库结构：

- **algorithm文件夹**：存放原项目中和算法有关的代码

- **data文件夹**：
    - original存放从比赛官网上下载来的初始数据
    - processed存放对初始数据处理后的数据，用于debug
- **output文件夹**：存放所有对因果图的预测结果，可以输入compare.py进行评分
- **compare.py**：songhao提供的评分程序
- **demo_causal_prior.py**：
    - 主程序，用于生成对因果图的预测
    - 其中CBNB_w、NBCN_w带有先验知识
    - 其中CBNB_e、NBCN_e貌似有BUG跑不通
    - 其他算法不带有先验知识
- **utils.py**：一些工具函数
- **环境配置.md**：如何配置环境的说明
- **论文解读.md**：论文的阅读笔记
- **实验结果.md**：所有算法的参数即得分