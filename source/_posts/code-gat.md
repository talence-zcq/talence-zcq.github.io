---
title: 代码精读<一>  GAT源码分析
date: 2022-12-03 11:28:35
tags: 代码精读
categories: 研0
mathjax: true
---

# GAT核心公式讲解
### 整体公式
公式中：a、W是要训练的参数
$$
a_{ij} = softmax(LeakyReLU(a^{-T}[W\vec h_i || W \vec h_j]))
$$

### 多头注意力
本文中实现的多头注意力公式如下
$$
\vec h_i' = ||_{k=1}^K \sigma( {\sum} _{j\in N_i}a _ {ij} ^k W^k \vec h^i)h_i'
$$

# 源代码讲解
### GAT网络各参数定义
``` python
class GATConv(nn.Module):
    def __init__(
        self,
        # 输入特征
        in_feats,
        # 输出特征
        out_feats,
        # 多头注意力的头数
        num_heads=1,
        # 特征dropout
        feat_drop=0.0,
        # 注意力dropout
        attn_drop=0.0,
        # 边dropout
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        # 出发点的输入特征，到达点的输入特征
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)

        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        # 用于设置全连接层
        if isinstance(in_feats, tuple):
            # 出发点的全连接层
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            # 到达点的全连接层
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            # 公用的全连接层
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            # 定义注意力权重方便之后训练
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        # 特征的dropout层
        self.feat_drop = nn.Dropout(feat_drop)
        assert feat_drop == 0.0 # not implemented
        # 注意力的dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        assert attn_drop == 0.0 # not implemented
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        # 初始化权重
        self.reset_parameters()
        self._activation = activation
```
<!---绿色--->
<div class="wg">
  <div class="t">
    提示
  </div>
  <div class="c">
    torch.nn.Parameter用于定义需要训练的参数。只需将网络中所有需要训练更新的参数定义为Parameter类型，再佐以optimizer，就能够完成所有参数的更新了。
  </div>
</div>
<br/>

### GAT网络初始化
#### 初始化层的重要性
适当的权值初始化可以加快模型的收敛，而不恰当的权值初始化可能引发梯度消失或者梯度爆炸，最终导致模型无法收敛。
#### 反向求梯度
下面的例子看完，你会对反向求梯度有更加深刻的理解   
考虑一个 3 层的全连接网络，公式如下：  
$$
H_1=X \times W_1 
$$

$$
H_2=H_1 \times W_2
$$

$$
Out=H_2 \times W_3
$$

其中第二层的网络梯度就可以推导如下：
$$
\triangle W_2 =  \frac{\delta Loss}{\delta W_2} = \frac{\delta Loss}{\delta Out} \times \frac{\delta Out}{\delta H_2} \times \frac{\delta H_2}{\delta W_2} = \frac{\delta Loss}{\delta Out} \times \frac{\delta Out}{\delta H_2} \times H_1
$$

所以$\triangle W_2$依赖于前一层的输出$H_1$ 。如果$H_1$趋近于零，那么$\triangle W_2$也接近于 0，造成梯度消失。如果$H_1$趋近于无穷大，那么$\triangle W_2$也接近于无穷大，造成梯度爆炸。要避免梯度爆炸或者梯度消失，就要严格控制网络层输出的数值范围。

#### GAT初始化代码
```Python
    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value
```
<!---绿色--->
<div class="wg">
  <div class="t">
    提示
  </div>
  <div class="c">
    nn.init.calculate_gain用来得到一个原有分布经过特定的激活函数后标准差变化的幅度<br/>
    nn.init.xavier_normal_(self.fc_src.weight, gain=gain)用来的到一个放大gain倍的正态分布初始化。<br/>
    以上所有初始化的目的都是为了能使每一层的网络输出的方差能够控制在1左右。
  </div>
</div>
<br/>

### GAT前向训练
```Python
    def forward(self, graph, feat, perm=None):
        # 只是为了计算权重，并不是真正改变图的属性
        with graph.local_scope():
            # 是否允许存在点的度为0
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            # 将特征通过全连接层,这里作者先做了全连接，之再做拼接
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            # 特征归一化
            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                # 对最后一维度求和，但是不改变结果维度
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                # 这里用到了dgl的内部消息传递机制
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                if perm is None:
                    perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)
            return rst

```
<!---绿色--->
<div class="wg">
  <div class="t">
    提示
  </div>
  <div class="c">
   这里作者对原本的GAT做了改动，使得复杂度降低了。具体改动为，将全连接运算和注意力运算的先后换了一下位置。
  </div>
</div>
<br/>

# 个人感悟
代码需要一定的pytorch和dgl基础，但未来很可能是pyg和dgl二分天下的时代，需要尽快熟悉这两种框架。