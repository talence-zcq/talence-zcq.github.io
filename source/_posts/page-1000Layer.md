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
| 数据集(RevGNN-Wide)| ROC-AUC | Mem | Params|
| ----------- | ----------- | ----------- | ----------- |
| ogbn-proteins | $88.24±0.15$ | $7.91$ | $68.47M$ |
| ogbn-arxiv| $74.05 ± 0.11$  | $8.49$ | $3.88M$ |  


# 创新点概要

### 特征矩阵三合一  
作者认为点和边在特征矩阵上会有一定联系，为了捕获每个维度的特征。因此将点特征矩阵（X∈N×D）、邻接矩阵（A∈N×N）、边特征矩阵（E∈M×F）映射到 X' 特征矩阵上（N×D），公式如下:
$$
f_w = X×A×U➡X'
$$


<!---黄色-->
<div class="wy">
  <div class="t">
    注意
  </div>
  <div class="c">
    这里的X是每一层网络的输入。A和E的特征矩阵在每一层都是一致的，都是初始值。
  </div>
</div>
<br/>

### 分组可逆图卷积


<br/>  

# 个人心得