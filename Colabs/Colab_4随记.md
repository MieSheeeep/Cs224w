

---

**同构图 vs 异构图**

homogenous graphs  同构
heterogeneous graphs 异构

同构图里，所有节点都是同一种类型，所有边也是同一种类型。比如一个社交网络里，节点全是"用户"，边全是"好友关系"。
异构图则允许节点和边有不同的类型。比如一个学术网络里，节点可以是"作者"、"论文"、"会议"三种类型，边可以是"撰写了"、"发表于"、"引用了"等不同类型。

**异质消息传递是什么意思**

在同构GNN里，所有邻居发来的消息都用同一套函数来计算和聚合。但在异构图里，不同类型的关系（比如"作者→论文"和"论文→会议"）语义完全不同，用同一套参数处理显然不合理。

所以异质消息传递的做法是：**为每一种边类型（或者说每一种关系）分别定义独立的消息传递函数**。比如"作者→撰写→论文"这条边用一组参数，"论文→引用→论文"这条边用另一组参数。每种关系有自己的 message function 和 aggregation，最后再把来自不同关系类型的聚合结果合并到目标节点上。

简单说就是：**关系类型不同，处理方式就不同**，而不是一刀切地用同一套变换处理所有邻居消息。这样模型就能捕捉到不同类型关系各自的语义特征。

---

**节点类型与节点特征分配**

DeepSNAP 框架把 NetworkX 图转换成它自己认识的异构图时，需要图上的每一个节点（Node）和每一条边（Edge）都已经挂上特定的“属性身份证”。对于节点来说，它强制要求提供三个属性：

* `node_type`（字符串类型）：告诉框架这个节点属于什么异构类型，决定了异质消息传递（Heterogenous Message Passing）的路径。
* `node_label`（整数类型）：也就是节点分类任务的预测目标（Ground Truth）。
* `node_feature`（张量类型）：输入给图神经网络用来做计算的初始特征向量。

> **我的疑问：**
> nx.set_node_attributes！！ 可是它和上面那个函数不是很重叠么，作用相同了，标记类似了？而且有没有必要拆解一份底层代码我理解一下 

**解答与底层拆解：**
从数值上看，它们确实对应了同一套 `community_map`（`0` 和 `1`），感觉是把同一个信息存了两遍。但是在图神经网络（GNN）尤其是在 DeepSNAP 这个库的设计里，这两个属性承担着完全不同的架构角色，它们必须被拆分开：

1. **`node_type`（节点类型）是用来“决定网络结构”的。** 在异构图中，不同的 `node_type` 意味着它们可能拥有不同的特征维度，或者需要经过不同的神经网络权重层处理。
2. **`node_label`（节点标签）是用来“计算损失（Loss）预测的答案”的。** 只有在计算成绩对错时，才会看 `node_label`，在前向计算中毫无作用。

*为什么这个作业会有这种重叠感？* 因为它为了让你快速在同一张图上练习，把空手道俱乐部的“派系(0和1)”既当成结构上的不同类型，又当做最终要预测的任务标签。在真实世界里，`node_type` 可能是“人”、“商品”，而 `node_label` 可能是“购买(1) / 不买(0)”，它们完全不会重合。

**底层伪代码理解：**
```python
def forward(self, x, node_type):
    # 1. 结构路由（因为 node_type 不同，经过的神经网络层也不同）
    if node_type == 'n0':
        out = self.linear_layer_for_n0(x)  # Mr. Hi 的专属处理
    elif node_type == 'n1':
        out = self.linear_layer_for_n1(x)  # Officer 的专属处理
    return out

def compute_loss(self, outputs, node_labels):
    # 2. 只有在计算成绩对错时，才会看 node_label
    return F.cross_entropy(outputs, node_labels)
```

---

**边类型分配与一个异构图小陷阱**

```python
def assign_edge_types(G, community_map):
    edge_types_dict = {}
    for u, v in G.edges():
        if community_map[u] == community_map[v]:
            edge_types_dict[(u, v)] = f"e{community_map[u]}" # 同一个社团
        else:
            edge_types_dict[(u, v)] = 'e2' # 跨社团
            
    nx.set_edge_attributes(G, edge_types_dict, 'edge_type')
```

**⚠️ 异构图（Heterogeneous Graph）里的小陷阱：区分 `edge_type` 和 `message_type`。**
在 DeepSNAP 异构图中：
* `edge_type` 只代表边本身的类型（比如我们的 `'e0', 'e1', 'e2'`）。
* `message_type` 代表的是**信息传递的路径类型**，它是一个三元组 `(源节点类型, 边类型, 目标节点类型)`。比如从 `n0` 经过 `e0` 发向 `n0` 的消息，它的 `message_type` 就是 `('n0', 'e0', 'n0')`。这在异构图神经网络聚合信息时非常关键。

---

**懵逼与开悟时刻：图机器学习的数据表示**

> **我的真实感受：**
> 哇我现在很懵逼，怎么图机器学习中有这么多乱七八糟的数据表示格式，这又导致了它们在Pytorch中流动的时候我不知道它们是怎么传递的，又为了快速计算，合并了什么图，什么节点，怎么做batch，张量在哪一个方向上扩张，这令我感到不安，我在cv中是可以很好的捕捉到每一步计算在做什么的，但是在图机器学习中却不行 

**不安的根源与解答：**
为什么 CV 让你安心？因为图像数据是"规则"的。一张图是 `[C, H, W]`，一个 batch 就是 `[B, C, H, W]`。
图是"不规则"的，每个图节点数不同，每个节点的邻居数也不同，没办法简单地在某个维度上堆叠成一个整齐的大张量。图 ML 被迫发明了一套替代方案（这也是让人懵的原因），核心就三件事：
1. **特征 `x`** 矩阵形状是 `[总节点数, 特征维度]`。
2. **结构 `edge_index`** 是一个 `[2, 总边数]` 的矩阵，本质上就是邻接矩阵的稀疏表示。
3. **batching** 的做法是把多个小图"拼"成一个大图。batch 之后 `x` 的形状是 `[所有图节点总和, 特征维度]`，用一个 `batch` 向量告诉你每个节点属于哪个原始图。

> **追问：**既然输入的张量是固定的，那么如果有的就是填充不成输入的维度呢 

CV 里有这个问题，但图 ML 里根本不存在这个问题！
因为 GNN 的可学习参数（比如 `W`）只和特征维度 `F` 有关，和节点数量 `N` 无关。不管你的图有3个节点还是3万个节点，同一个权重矩阵 `W` 对每个节点独立地做 `x @ W`。真正需要固定的只有特征维度 `F`。

---

**Heterogeneous GNN Layer (异构图卷积层)**

核心思想：每种关系类型有自己独立的一套参数。
消息类型 $m=(s, r, d)$ 就是关系三元组。以节点 $v$（类型为 $d$）为例，更新公式：
$$h_v^{(l)[m]} = W^{(l)[m]} \cdot \text{CONCAT} \Big( W_d^{(l)[m]} \cdot h_v^{(l-1)}, W_s^{(l)[m]} \cdot AGG(\{h_u^{(l-1)}, \forall u \in N_{m}(v) \})\Big)$$

**公式与代码逐步对应（数据流从头到尾）：**

| 公式步骤 | 代码实现 | 维度变化 | 意义 |
| :--- | :--- | :--- | :--- |
| **第一步：AGG** | `out = matmul(edge_index, node_feature_src)` | `[N_src, F_src]` → `[N_dst, F_src]` | 对应 `message_and_aggregate`，聚合源节点邻居特征求均值 |
| **第二步：$W_s \cdot AGG$** | `src_out = self.lin_src(aggr_out)` | `[N_dst, F_src]` → `[N_dst, out]` | $W_s$ 处理邻居聚合结果 |
| **第二步：$W_d \cdot h_v$** | `dst_out = self.lin_dst(node_feature_dst)` | `[N_dst, F_dst]` → `[N_dst, out]` | $W_d$ 处理目标节点自身特征 |
| **第三步：CONCAT** | `concat_out = torch.cat([dst_out, src_out], dim=-1)` | → `[N_dst, out*2]` | 把自身和邻居信息拼接 |
| **第四步：$W$** | `aggr_out = self.lin_update(concat_out)` | → `[N_dst, out]` | 最外层的变换 $W$ |

就是公式从里到外，代码从上到下，完全一一对应，就这么一条线，没有什么神秘的。

---

**Heterogeneous GNN Wrapper Layer (消息合并)**

上一层针对每种关系类型分别算出了一个结果，现在要把它们合并成每个节点的最终表示。假设"论文"节点同时收到了"作者撰写"和"论文引用"两种消息，怎么合成一个？

* **方法一：均值聚合**
    最简单粗暴，直接取平均。$$h_v^{(l)} = \frac{1}{M}\sum_{m=1}^{M}h_v^{(l)[m]}$$
    这意味着模型认为不同关系对节点同等重要。
* **方法二：语义级注意力（HAN）**
    让模型自己学权重。
    1. 算每种消息类型的重要性得分 $e_m$（一个标量，代表整种关系类型的重要性，所有目标节点求平均得到）。
    2. softmax 归一化得到权重 $\alpha_m$。
    3. 加权求和：$$h_v^{(l)} = \sum_{m=1}^{M} \alpha_{m} \cdot h_v^{(l)[m]}$$
    注意：注意力权重 $\alpha_m$ 是**所有目标节点共享**的，这是一种粗粒度的注意力，计算效率高。

---

**初始化异构图神经网络层代码讲解**

自动获取不同边两端节点的特征维度逻辑：
```python
def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    convs = {}
    # 遍历 DeepSNAP 图结构中所有的消息类型 三元组
    for message_type in hetero_graph.message_types:
        if first_layer:
            # 第一层：读原始节点的特征维度
            in_channels_src = hetero_graph.num_node_features(message_type[0])
            in_channels_dst = hetero_graph.num_node_features(message_type[2])
        else:
            # 如果不是第一层，由于经过了前面的映射，维度都变成了 hidden_size
            in_channels_src = hidden_size
            in_channels_dst = hidden_size
        
        # 实例化 HeteroGNNConv 并用 message_type 作为 key 保存
        convs[message_type] = conv(in_channels_src, in_channels_dst, out_channels=hidden_size)
    return convs
```

**网络 Forward 和 Loss 怎么写：**

在 Forward 中，输入是一个含有多种节点和边的 **字典（Dict）**。不应该使用 `deepsnap.hetero_gnn.forward_op` 处理整个网络，而是直接调用 wrapper，遇到需要按类型分别处理的单体操作（如 BN, ReLU, Linear）才用 `forward_op`：
```python
def forward(self, node_feature, edge_index):
    x = node_feature
    # 1. 经过 wrapper 得到新 dict，单体操作用 forward_op 遍历
    x = self.convs1(x, edge_index)
    x = forward_op(x, self.bns1)
    x = forward_op(x, self.relus1)
    # ... 第二层同理 ...
    x = forward_op(x, self.post_mps) # 输出层
    return x
```

计算 Loss 时也要遍历 dict，且只拿 indices 里切分好的（Train/Val/Test）节点算：
```python
def loss(self, preds, y, indices):
    loss = 0
    loss_func = F.cross_entropy
    for node_type in preds:
        # 仅取出掩码 mask / indices 中有的节点计算交叉熵损失
        idx = indices[node_type]
        loss += loss_func(preds[node_type][idx], y[node_type][idx])
    return loss
```