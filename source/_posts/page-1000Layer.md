---
title: 论文笔记<二>  Training Graph Neural Networks with 1000 Layers
date: 2022-11-28 22:49:29
tags: 图神经网络
categories: 研0
mathjax: true
---

# 基本信息
### 作者
 **Guohao Li**  Intel Labs <guohao.li@kaust.edu.sa>  
 **Matthias Muller**  
 **Bernard Ghanem**   
 **Vladlen Koltun**  

 ### 发布会议
 Accepted at ICML 2021
 
 ### 数据集
| 数据集（RevGNN-Wide）| ROC-AUC | Mem | Params|
| ----------- | ----------- | ----------- | ----------- |
| ogbn-proteins | $88.24 ± 0.15$ |$7.91$|$68.47M$|
| ogbn-arxiv| $74.05 ± 0.11$  |$8.49$|$3.6\%$|

# 创新点概要
 ### 分组可逆卷积
$$
X_0' = \sum_{i=2}^{C}X_i
$$

$$
X_i' = f_{w_i}(X_{i-1}',A,U)+X_i, i\in \{1,...,C\}
$$

$$
X_i = X_i'-f_{w_i}(X_{i-1}',A,U),i\in\{2,...,C\}
$$

$$
X_0'=\sum_{i=2}^CX_i
$$

$$
X_1 = X_1'-f_{w1}(X_0',A,U)
$$
 ### 权重共享
代码讲解
 ### 状态平衡
$$
Z^* = f_w^{DEQ}(Z^*,X,A,U)
$$

$$
Z'=GraphConv(Z_{in},A,U)
$$

$$
Z''= Norm(Z'+X)
$$

$$
Z''' = GraphConv(Dropout(ReLU(Z'')),A,U)
$$

$$
Z_o = Norm(ReLU(Z'''+Z'))
$$
 