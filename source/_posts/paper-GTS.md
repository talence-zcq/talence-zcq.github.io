---
title: 论文笔记<一> DISCRETE GRAPH STRUCTURE LEARNING FOR FORECASTING MULTIPLE TIME SERIES
date: 2022-11-27 15:20:38
tags: 时空数据预测
categories: 时空数据预测
mathjax: true
---

# 基本信息
### 作者
**Chao Shang∗** University of Connecticut chao.shang@uconn.edu   
**Jie Chen†** MIT-IBM Watson AI Lab, IBM Research chenjie@us.ibm.com  
**Jinbo Bi** University of Connecticut jinbo.bi@uconn.edu
### 发布会议
Published as a conference paper at ICLR 2021
### 数据集
| 数据集| MAE | RMSE | MAPE|
| ----------- | ----------- | ----------- | ----------- |
| METR-LA | $2.99$ |$5.85$|$8.3\%$|
| PEMS-BAY| $1.58$  |$3.30$|$3.6\%$|
| PMU | $0.41*10^{-3}$ |$0.30*10^{-2}$|$0.07\%$|   

<br/>
<br/>
 
# 创新点概要
### 通过数据生成图结构
作者认为，很多情况下，人们并不能获得完整的图结构信息。因此，当**没有图的结构信息**或者**只有部分图结构信息**时，需要通过数据来对图的信息进行获取。具体操作如下：

 首先，在时空数据中，X张量具有三个维度<特征(f)，时间(t)，批量(i)>。作者将**时间维度**进行压缩，压缩成<特征，批量>。具体公式如下：
$$
MLP(Vec(Conv(X_i)))
$$

<!---黄色-->
<div class="wy">
  <div class="t">
    注意
  </div>
  <div class="c">
    MLP代表全连接层，Vec代表将多维张量将为成一维向量，Conv代表卷积抽取特征
  </div>
</div>

<br/>

在我们得到了<特征，批量>的二维张量$z_i$之后。将其两两配对，通过拼接后丢入两个全连接层，得到一个概率θ，具体公式如下：
$$
MLP(MLP(z_i))
$$


两两节点生成一个概率θ，对应的便是这两个点在邻接矩阵中是1的概率。我们自然可以设定一个阈值，大于这个阈值的概率视为1，否则视为0。于是问题来了，你如何在这一步将梯度传回给之前的MLP和Conv操作呢？因为阈值操作本质是比大小，没办法求导。这时，就需要一个Trick——Gumbel reparameterization。

### Gumbel reparameterization
由于我们的值域已经确定，为{0，1}，代表两点之间的连接存在与否。现在需要做的就是依据给定概率θ，在这个范围中采样。要使采样的过程可导，只需要把阈值的操作替换成Gumbel softmax方法即可，具体公式如下：
$$
out = softmax(log(p)+G) 
$$
<!---黄色-->
<div class="wy">
  <div class="t">
    注意
  </div>
  <div class="c">
    p为输入概率，如[0.6，0.4]。out为输出概率，如[0.8，0.2]，将out转化为one-hot向量，与值域{0，1}相乘，即可得到采样结果
  </div>
</div>
<br/>

得到采样后的邻接矩阵，就可以将其当作真实的邻接矩阵，放入T-GCN之类的图卷积网络中进行特征运算了。此时的损失函数为：
$$
l^t_{base} ={1\over num} \sum|X_{pre}-X_{true}|
$$
<br/>

### 先验图结构的价值
但是，作者想到，万一给定的部分图信息是真实可信的，为了使生成图结构的结果不过分偏离给定图，作者又设置了如下损失函数：

$$
l_{reg}=\sum -A_{ij}*logθ_{ig}-(1-A_{ij})*log(1-θ_{ij})
$$

<!---绿色--->
<div class="wg">
  <div class="t">
    提示
  </div>
  <div class="c">
    这里就是惩罚和给定图结构不相符的邻接点，越不相符，值越大
  </div>
</div>
<br/>
然后总损失函数就得到了：

$$
l=l_{base}+\lambda l_{reg}
$$

### 总结
最后放上该模型的总概览图：
![](https://zcq-hexo.oss-cn-hangzhou.aliyuncs.com/img/20221127191548.png)
<br/>
 
# 个人心得
1. 之后的图卷积，若使用transformer而不是GRU，可能会取得更好的结果
2. 感觉单纯的把时间全部压缩在一起来预测两点之间的连接概率有些草率。
3. 如果能从实际规律出发，找到更好的时间与空间的联系，来进行两点间概率预测。也许会取得更好的结构预测结果