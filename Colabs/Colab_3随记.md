## 消息传递范式（Message Passing）

几乎所有 GNN 都遵循同一个框架：

1. **Message**：每个节点从邻居那里收集信息
2. **Aggregate**：把收到的所有消息汇总成一个向量
3. **Update**：用汇总后的信息更新自身表示

在 PyG（PyTorch Geometric）中，这个框架由 `MessagePassing` 基类实现。用户只需重写 `message()`、`aggregate()`、`update()` 方法，然后调用 `propagate()` 即可自动执行完整的消息传递流程。



------

#### `message` 方法：为每条边生成消息

```python
def message(self, x_j):
    # x_j 是邻居的特征（PyG 的 _j 后缀 = 源节点/邻居）
    return self.lin_r(x_j)  # W_neigh · h_j
```

##### 1. 它的输入 / 输出是什么？

- 输入 `x_j`

  ：PyG 会自动把「每条边的源节点（邻居节点）特征」传给这个参数。

  

  比如边 

  ```
  (j → i)
  ```

  （从邻居 j 到目标节点 i），

  ```
  x_j
  ```

   就是节点 j 的特征向量。

- **输出**：这条边要传递的「消息」，也就是邻居节点 j 发送给目标节点 i 的信息。

##### 2. 它到底在做什么？

- 对**图中的每一条边**，取出邻居节点（源节点）的特征 `x_j`，
- 用一个线性层 `self.lin_r`（也就是代码注释里的 `W_neigh`）做一次特征变换，
- 把变换后的结果作为 “消息”，发送给目标节点。

举个例子：如果节点 j 的特征是 `[0.1, 0.2, 0.3]`，经过 `lin_r` 变换后变成 `[0.5, 0.6]`，那 `[0.5, 0.6]` 就是 j 发送给目标节点的消息。

------

#### `aggregate` 方法：把同一目标节点的消息汇总

```python
def aggregate(self, inputs, index, dim_size=None):
    # scatter 按目标节点编号分组，取均值
    return torch_scatter.scatter(inputs, index, dim=self.node_dim,
                                 dim_size=dim_size, reduce='mean')
```

##### 1. 它的输入 / 输出是什么？

- **输入 `inputs`**：`message` 方法为所有边生成的消息集合，是一个和边数等长的张量。
- **输入 `index`**：每条边对应的「目标节点索引」，用来标记 “这条边的消息要发给哪个节点”。
- **输出**：每个目标节点收到的所有消息的聚合结果，是一个和节点数等长的张量。

##### 2. 它到底在做什么？

用 `torch_scatter.scatter` 实现「按目标节点分组 + 聚合」：

1. **分组**：根据 `index`，把所有消息按目标节点编号分组（比如所有发给节点 i 的消息会被分到同一组）。
2. **聚合**：对每组消息执行 `reduce='mean'`（取均值），把多个邻居发来的消息压缩成一个向量，作为目标节点的聚合消息。

举个例子：节点 i 收到 3 条邻居消息 `[0.5, 0.6]`、`[0.7, 0.8]`、`[0.9, 1.0]`，`aggregate` 会计算它们的均值 `[(0.5+0.7+0.9)/3, (0.6+0.8+1.0)/3] = [0.7, 0.8]`，作为节点 i 的聚合消息。

------

## GraphSage（Graph Sample and Aggregate）

### 来源与动机

- **论文**：*"Inductive Representation Learning on Large Graphs"*（Hamilton, Ying, Leskovec, 2017, Stanford）
- **核心动机**：解决 GCN 的**直推式（transductive）**限制

GCN 的问题在于它是直推式的——训练时需要看到整个图（包括测试节点）。如果图中新增节点，必须重新训练整个模型。这在动态场景（如社交网络新用户、推荐系统新商品）中不可行。

GraphSage 的核心思想：**不为每个节点学一个固定 embedding，而是学习一种"如何从邻居提取信息"的策略（聚合函数）**，从而天然支持新节点的归纳推理（inductive）。

### 核心机制

每一层做三步：

**第一步：邻居聚合（Aggregate）**

从节点 v 的邻居集合 N(v) 中收集特征，用聚合函数压缩成一个向量：

$$h_{N(v)} = \text{AGG}({ h_u : u \in N(v) })$$

常见的 AGG 有 mean（取均值）、max-pool、LSTM 等。最简单且最常用的是 mean。

**第二步：拼接（Concatenate）**

把聚合得到的邻居信息和节点自身的特征拼接：

$$\text{concat}(h_v, h_{N(v)})$$

**第三步：线性变换 + 非线性激活**

$$h_v' = \sigma(W \cdot \text{concat}(h_v, h_{N(v)}))$$

### 拼接的意义

> **我的初始理解**：GraphSage 就是把原本的特征做一个聚合之后得到的通道直接与原通道做拼接，然后线性变换为原通道维度。也就是说它提取到了一些聚合信息？
>
> **深入理解**：方向是对的，但可以更精确。

- **$h_v$** 代表"**我自己是什么**"——节点自身的特征
- **$h_{N(v)}$** 代表"**邻居们在说什么**"——节点周围局部结构的摘要

这两者携带的信息本质上不同。如果只用聚合信息，就丢掉了节点本身的身份；如果只用自身信息，就没有利用图结构。拼接的意义在于**让模型同时看到这两路信号，由后面的线性变换 W 来学习如何融合它们**。

**举例**：在社交网络中，节点 v 是一个程序员（自身特征：职业、年龄、技能），他的邻居大多是设计师（聚合特征：设计相关特征偏多）。拼接之后，模型可以学到"这是一个**和设计师群体关系密切的**程序员"，比单独看任何一路信息都更丰富。

### 与 GCN 的对比

GCN 的更新公式：

$$h_v' = \sigma\left(W \cdot \sum_{u \in N(v) \cup {v}} \frac{h_u}{\sqrt{d_u \cdot d_v}}\right)$$

GCN 把自身信息也混进求和里（通过加自环），然后统一做一次变换。**自身信息和邻居信息共享同一个权重矩阵**。

GraphSage 通过拼接，等价于给自身和邻居**分配了不同的权重**。把 W 按拼接维度拆开：

$$W \cdot \text{concat}(h_v, h_{N(v)}) = W_{\text{self}} \cdot h_v + W_{\text{neigh}} \cdot h_{N(v)}$$

模型可以用 $W_{\text{self}}$ 控制"保留多少自身信息"，用 $W_{\text{neigh}}$ 控制"吸收多少邻居信息"，两者解耦，灵活性更大。

### 维度变化

假设输入特征维度为 d，聚合后的邻居特征也是 d 维：

```
h_v      : [d]       ← 自身
h_N(v)   : [d]       ← 邻居聚合
concat   : [2d]      ← 拼接
W        : [d, 2d]   ← 线性变换
output   : [d]       ← 变换回原维度
```

拼接让输入维度暂时翻倍，W 矩阵再把它压回去，这个过程中模型学会了如何取舍和融合两路信息。

### 多层堆叠的效果

堆叠多层 GraphSage 时，信息传播的范围逐步扩大：

- **第 1 层**：每个节点看到自己的 1-hop 邻居
- **第 2 层**：每个节点间接看到 2-hop 邻居（邻居在上一层已经聚合了它们的邻居）
- **第 k 层**：感受野扩展到 k-hop

像一个逐步扩大的"信息气泡"，每一层都在更大的范围内提取结构信息，同时始终保留节点自身的身份。

### 代码实现解析

```python
class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, ...):
        # lin_l: 自身特征的线性变换 W_self
        self.lin_l = Linear(in_channels, out_channels, bias=bias)
        # lin_r: 邻居特征的线性变换 W_neigh
        self.lin_r = Linear(in_channels, out_channels, bias=bias)
```

**forward 方法**：

```python
def forward(self, x, edge_index, size=None):
    # 1. propagate 触发 message → aggregate 流程
    #    x=(x, x) 表示源节点和目标节点用同一组特征
    out = self.propagate(edge_index, x=(x, x), size=size)
    # 2. Skip connection: W_self · h_i + AGG(W_neigh · h_j)
    out = self.lin_l(x) + out
    # 3. L2 归一化
    if self.normalize:
        out = F.normalize(out, p=2, dim=-1)
    return out
```

**message 方法**（对每条边调用）：

```python
def message(self, x_j):
    # x_j 是邻居的特征（PyG 的 _j 后缀 = 源节点/邻居）
    return self.lin_r(x_j)  # W_neigh · h_j
```

**aggregate 方法**（把同一目标节点的所有消息汇总）：

```python
def aggregate(self, inputs, index, dim_size=None):
    # scatter 按目标节点编号分组，取均值
    return torch_scatter.scatter(inputs, index, dim=self.node_dim,
                                 dim_size=dim_size, reduce='mean')
```

###### 1. 形象比喻：快递分拣中心

想象一下，你有很多快递（`inputs`），每个快递都要发往特定的城市（`index`）：

- **`inputs`**：邻居节点传过来的特征向量（快递的内容）。
- **`index`**：目标节点的编号（快递单上的目的地城市 ID）。
- **`scatter`**：分拣员。他看到快递 A 要去城市 1，快递 B 要去城市 2，快递 C 也要去城市 1。
- **`reduce='mean'`**：加工方式。分拣员把发往同一个城市的所有快递拆开，把里面的东西混合在一起，取个平均值。

------

###### 2. 数值例子演示

假设我们有 3 条边，分别指向节点 0 和节点 1：

- 边 1：从邻居 A → 目标 **0**，特征是 `[2, 2]`
- 边 2：从邻居 B → 目标 **0**，特征是 `[4, 4]`
- 边 3：从邻居 C → 目标 **1**，特征是 `[10, 10]`

那么对应的输入是：

- `inputs = [[2, 2], [4, 4], [10, 10]]`
- `index  = [0, 0, 1]` （前两个给 0 号，后一个给 1 号）

执行 `scatter(..., reduce='mean')` 后的结果：

- **节点 0**：$( [2, 2] + [4, 4] ) / 2 = \mathbf{[3, 3]}$
- **节点 1**：$ [10, 10] / 1 = \mathbf{[10, 10]}$

最终输出 `out` 就是 `[[3, 3], [10, 10]]`。

------

###### 3. 参数详解

- **`inputs`**：这是所有“口信”（Message）。在 PyG 的 `propagate` 流程中，通常是邻居节点的特征。
- **`index`**：这是一个索引数组，长度和 `inputs` 的第一维一样。它告诉程序：第 $i$ 个特征应该归属于哪个目标节点。
- **`dim`**：在哪一维进行聚合。通常 `dim=0`（行方向）。
- **`dim_size`**：输出的大小。通常等于图中节点的总数。这保证了即使某个节点没有邻居（孤立点），输出结果也会给它留个位置（通常填充 0）。
- **`reduce='mean'`**：聚合策略。GraphSAGE 最经典的做法就是 `mean`。你也可以换成 `sum` 或 `max`。



**完整公式**：

$$h_i' = W_l \cdot h_i + \text{MEAN}({ W_r \cdot h_j : j \in N(i) })$$

这与 `W · concat(h_i, AGG(h_j))` 表达能力等价——对 concat 做一次线性变换，等价于分别做两个线性变换再相加。代码实现直接用了后者，省去了拼接的步骤。

### 关于 x=(x, x) 的含义

表示源节点和目标节点使用**同一组特征**。PyG 支持二部图（bipartite graph），因此接口允许传入两组不同特征。在普通图（同构图）上，源和目标是同一批节点，所以传 `(x, x)`。
理解 GraphSAGE 的难点在于：**数学公式写得非常“数学”，而代码实现为了效率，利用了矩阵运算的性质把步骤“折叠”了。**

咱们把公式和代码像拼图一样对齐，你就能看清那“三步”到底藏在哪了。

------

## GAT（Graph Attention Network）

### 具体计算过程

**第一步：线性变换**

$$z_i = W \cdot h_i$$

W 是所有节点共享的权重矩阵，这一步与邻居无关。

**第二步：计算注意力分数（逐边）**

$$e_{ij} = \text{LeakyReLU}(a^T \cdot \text{concat}(z_i, z_j))$$

a 是一个可学习的向量。把节点 i 和 j 变换后的特征拼起来，点乘 a，过 LeakyReLU。

关键点：**a 向量对所有边都是同一个**。模型学到的不是"节点 3 对节点 7 的权重是多少"，而是一种**通用的评判标准**——"什么样的节点对之间应该有更强的信息传递"。

**第三步：softmax 归一化**

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(i)} \exp(e_{ik})}$$

让节点 i 的所有邻居的权重加起来等于 1。邻居是 3 个就在 3 个数上做 softmax，邻居是 300 个就在 300 个数上做 softmax。

**第四步：加权聚合**

$$h_i' = \sigma\left(\sum_{j \in N(i)} \alpha_{ij} \cdot z_j\right)$$

### 具体图例

用一个 5 节点图（A-E）走一遍完整计算。假设每个节点有 2 维特征：A=[1,2], B=[3,1], C=[0,4], D=[2,3], E=[1,0]。

以**节点 A**（邻居为 B, C, D）为例：

```
Step 1: 对每条边计算原始注意力分数
  e_AB = a^T · concat(z_A, z_B) = 1.2
  e_AC = a^T · concat(z_A, z_C) = 0.5
  e_AD = a^T · concat(z_A, z_D) = 2.1

Step 2: softmax 归一化（在 3 个数上做）
  α_AB = exp(1.2) / (exp(1.2)+exp(0.5)+exp(2.1)) = 0.254
  α_AC = exp(0.5) / ... = 0.126
  α_AD = exp(2.1) / ... = 0.620
  总和 = 1.000

Step 3: 加权聚合
  h_A' = 0.254 × z_B + 0.126 × z_C + 0.620 × z_D
```

以**节点 B**（邻居只有 A, C）为例：

```
Step 1:
  e_BA = 0.8
  e_BC = 1.6

Step 2: softmax（只在 2 个数上做）
  α_BA = 0.310
  α_BC = 0.690
  总和 = 1.000

Step 3:
  h_B' = 0.310 × z_A + 0.690 × z_C
```

**关键观察**：A 有 3 个邻居，B 只有 2 个，但计算过程完全一样——因为注意力函数每次只看一对节点，softmax 自适应邻居数量。

### 个人疑问

#### GAT to GraphSage？

> **我的思路**：GAT 是不是就是在 GraphSage 的基础上，对每个邻居的权重使用 Attention 层？GraphSage 用 mean 对所有邻居一视同仁，而 GAT 让模型学习每个邻居该分多大的权重。
>
> **确认**：是的。GraphSage 的 mean 聚合是一种"民主投票"——每个邻居的贡献完全平等。GAT 把这变成了"加权投票"——模型自动判断哪些邻居更重要。

#### **核心疑问：邻居数量不同，怎么做 Attention？**

> **我的疑问**：可是每个节点的邻居不都是不同的吗，并且邻居数量也不同，如何使用 Attention 呢？
>
> **关键理解**：GAT 的 Attention **不依赖于邻居数量**。

注意力函数只关心**一对节点**：

$$e_{ij} = a(\mathbf{W} \cdot h_i, \mathbf{W} \cdot h_j)$$

a 是一个共享的、只接受**两个向量**作为输入的函数。不管节点 i 有 3 个邻居还是 300 个邻居，对每一条边 (i, j)，都是用同一个函数独立地算出一个标量分数。

然后在节点 i 的所有邻居上做 **softmax 归一化**：

$$\alpha_{ij} = \text{softmax}*j(e*{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in N(i)} \exp(e_{ik})}$$

**softmax 天然支持不同长度的集合**——它只是对一组数做指数归一化，不管这组数有多少个。整个过程中，没有任何矩阵的维度和邻居数量绑定。

#### **GAT 还算 Attention 吗？**

> **我的疑问**：GAT 舍弃了 Attention 的 QKV 格式，那么它还算 Attention 么？
>
> **理解**：QKV 只是 Attention 的**一种实现形式**，不是 Attention 的定义。

Attention 的本质定义：**根据输入内容动态计算权重，然后加权聚合**。只要满足这一点就是 Attention。

| 特性       | Transformer                  | GAT                                            |
| ---------- | ---------------------------- | ---------------------------------------------- |
| 注意力范围 | 全连接（每个位置和所有位置） | 稀疏（只在图的边上）                           |
| 计算形式   | Q·K^T（矩阵乘法）            | a^T · concat(z_i, z_j)（逐边标量）             |
| 计算量     | O(n²)                        | O(\|E\|)（边数）                               |
| 核心区别   | 在学"该关注**谁**"           | 已知可以关注谁（由边决定），在学"关注**多少**" |

QKV 是 Transformer 为了在长序列上高效并行计算而设计的形式。GAT 面对的是图结构，没有固定序列，所以用了一种更适合图的、逐边计算的形式。本质上做的是同一件事。

对比 GraphSage 的 mean 聚合——所有邻居权重固定为 1/|N|，这显然**不是** Attention。GAT 的权重由输入动态决定，所以是 Attention。

### 多头注意力

每个头有自己独立的 $W^k$ 和 $a^k$，独立学一套注意力模式：

$$h_i' = \text{concat}(\text{head}_1, \text{head}_2, \ldots, \text{head}_K)$$

$$\text{head}*k = \sigma\left(\sum*{j \in N(i)} \alpha_{ij}^k \cdot W^k \cdot h_j\right)$$

每个头输出维度为 `out_channels`，拼接后输出维度为 `heads × out_channels`。

#### 为什么 out_channels 不需要除以 heads？

> **我的疑问**：在代码中 `self.att_l = Parameter(torch.Tensor(heads, out_channels))`，为什么 out_channels 不用除以 heads？
>
> **理解**：这是 GAT 和 Transformer 在多头设计上的根本区别。

```
Transformer 的做法：总维度固定，按头数拆分
  d_model=512, heads=8 → 每个头 64 → 拼接后 512（回到原始维度）
  → 目的是保持输入输出维度一致

GAT 的做法：每个头维度固定，拼接后变大
  out_channels=64, heads=8 → 每个头 64 → 拼接后 512
  → out_channels 本身就是每个头的维度，不是总维度
```

**一句话总结**：Transformer 是"先定总预算再分给各头"，GAT 是"先定每个头的预算再加起来"。

这也直接解释了 GNNStack 中后续层输入维度要乘以 heads 的原因——第一层拼接后维度变成了 `heads * hidden_dim`，下一层的输入必须匹配。

#### GAT v1 的表达力局限——a 向量的问题

> **我的质疑**：a^T 是所有节点对共用的，我挺怀疑它是否能够包含这么多配对信息。这意味着 z 这些节点 hidden 特征需要尽量大。
>
> **深入分析**：这个怀疑是完全正确的，这确实是 GAT v1 的一个已知缺陷。

##### 问题分析

把注意力向量 a 拆成两半来看：

$$a^T \cdot \text{concat}(z_i, z_j) = a_{\text{left}}^T \cdot z_i + a_{\text{right}}^T \cdot z_j$$

注意力分数其实是**两个独立标量之和**——节点 i 和节点 j 之间**没有真正的交互项**。

更严重的是，做 softmax 时 $a_{\text{left}}^T \cdot z_i$ 对节点 i 的所有邻居是同一个常数，会被约掉。因此注意力排序**完全由 $a_{\text{right}}^T \cdot z_j$ 决定**。

这意味着：不管查询节点是谁，邻居之间的注意力排序都是固定的——这就是所谓的**静态注意力（static attention）**问题。注意力并不是真正"动态"的。

##### 缓解手段

- **多头注意力**：每个头有独立的 W 和 a，从不同角度评估邻居重要性，但每个头内部仍然是静态的
- **增大隐藏维度**：正如我所直觉到的，更大的 z 让 $a_{\text{right}}^T \cdot z_j$ 能编码更丰富的信息，使不同节点在标量投影上更容易区分开。但本质限制还在

##### GATv2 的修复

来自 2021 年论文 *"How Attentive are Graph Attention Networks?"*（Brody 等人），正式指出了此问题并提出修复：

```
GAT (v1):  e_ij = LeakyReLU( a^T · concat(W·h_i, W·h_j) )
                 = LeakyReLU( a_left^T·W·h_i + a_right^T·W·h_j )
                 → 两项独立，无交互

GATv2:     e_ij = a^T · LeakyReLU( W · concat(h_i, h_j) )
                 → 先拼接再非线性，W 矩阵产生 h_i 和 h_j 的交叉项
                 → 真正的动态注意力
```

区别很微妙——只是 LeakyReLU 和线性变换的顺序调换——但效果完全不同。GATv2 中 W 作用在拼接后的向量上，矩阵乘法天然产生 h_i 和 h_j 之间的交叉项，注意力变成了真正的动态注意力。

### 代码实现解析

#### 初始化

```python
class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2, ...):
        # 线性变换：输入维度 → heads * out_channels
        self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
        self.lin_r = self.lin_l  # 源节点和目标节点共享同一个线性变换

        # 注意力参数：每个头一个 out_channels 维的向量
        self.att_l = Parameter(torch.Tensor(heads, out_channels))  # 目标节点的 a_left
        self.att_r = Parameter(torch.Tensor(heads, out_channels))  # 源节点的 a_right
```

注意 `self.lin_r = self.lin_l`，说明对所有节点做的线性变换是相同的。注意力的区分完全由 `att_l` 和 `att_r` 承担。

#### forward 方法

```python
def forward(self, x, edge_index, size=None):
    H, C = self.heads, self.out_channels

    # 1. 线性变换，然后 reshape 成 [N, H, C]（多头维度展开）
    x = self.lin_l(x)        # [N, H*C]
    x = x.view(-1, H, C)     # [N, H, C]

    # 2. 计算 alpha：对每个头，做 z_i 与 att 向量的点积
    #    这就是 a_left^T · z_i 和 a_right^T · z_j 中的各自部分
    alpha_l = (x * self.att_l).sum(dim=-1)  # [N, H]，目标节点的注意力分量
    alpha_r = (x * self.att_r).sum(dim=-1)  # [N, H]，源节点的注意力分量

    # 3. 消息传递
    out = self.propagate(edge_index, x=(x, x),
                         alpha=(alpha_l, alpha_r), size=size)

    # 4. reshape 回 [N, H*C]（多头拼接）
    out = out.view(-1, H * C)
    return out
```

**维度追踪（假设 N=100, in=16, H=4, C=32）**：

```
x:       [100, 16]   → lin_l → [100, 128] → view → [100, 4, 32]
alpha_l: [100, 4]    （每个节点在每个头上的注意力分量）
alpha_r: [100, 4]
out:     [100, 4, 32] → view → [100, 128]
```

#### message 方法

```python
def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
    # 1. 注意力分数 = alpha_i + alpha_j（即 a_left^T·z_i + a_right^T·z_j）
    #    然后 LeakyReLU
    out = alpha_i + alpha_j          # [E, H]
    out = F.leaky_relu(out, self.negative_slope)

    # 2. softmax 归一化（按目标节点分组）
    out = softmax(out, index, ptr, size_i)  # [E, H]

    # 3. dropout
    out = F.dropout(out, p=self.dropout, training=self.training)

    # 4. 加权：注意力权重 × 邻居特征
    out = x_j * out.unsqueeze(-1)   # [E, H, C]
    return out
```

**关键细节**：

- `alpha_i` 和 `alpha_j` 是 PyG 根据 `edge_index` 自动从 `alpha` 元组中取出的——`_i` 是目标节点，`_j` 是源节点
- `softmax(out, index, ...)` 中的 `index` 是每条边对应的目标节点编号，实现了"在每个节点的邻居上分别做 softmax"
- `out.unsqueeze(-1)` 把 `[E, H]` 扩展为 `[E, H, 1]`，与 `x_j` 的 `[E, H, C]` 广播相乘

> **注意**：原始作业代码中 message 方法里写的是 `out = self.att_l * alpha_i + self.att_r * alpha_j`，但 `alpha_i` 和 `alpha_j` 已经是点积后的标量结果 `[E, H]`，不应该再乘以 `att_l`/`att_r`（形状 `[H, C]`）。正确的写法应该是 `out = alpha_i + alpha_j`。这可能是原始作业代码中的一个 bug。

#### aggregate 方法

```python
def aggregate(self, inputs, index, dim_size=None):
    # 用 sum 而非 mean（因为注意力权重已经归一化了）
    return torch_scatter.scatter(inputs, index, dim=self.node_dim,
                                 dim_size=dim_size, reduce='sum')
```

注意这里用 `reduce='sum'`，而 GraphSage 用 `reduce='mean'`。因为 GAT 的注意力权重经过 softmax 已经归一化（加起来为 1），加权求和本身就是一种加权平均。

------

## GNNStack：通用 GNN 骨架

> **我的疑问**：GNNStack 是在做什么？
>
> **理解**：它就是把多个 GNN 层堆叠起来，形成一个完整的网络。和多层 MLP 或多层 CNN 是一样的概念——单层看 1-hop，多层扩展感受野。通过 `build_conv_model` 切换内部用 GraphSage 还是 GAT，其他结构完全不变。

### 结构

```python
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        conv_model = self.build_conv_model(args.model_type)

        # 第一层：原始特征 → hidden_dim（每个头）→ 拼接后 heads * hidden_dim
        self.convs.append(conv_model(input_dim, hidden_dim))

        # 后续层：heads * hidden_dim → hidden_dim（每个头）→ 拼接后又是 heads * hidden_dim
        for l in range(args.num_layers - 1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))

        # 分类头：heads * hidden_dim → hidden_dim → output_dim
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim)
        )
```

### 维度变化示例

假设 `input_dim=16, hidden_dim=32, heads=4, num_layers=3, output_dim=5`：

```
输入特征:       [N, 16]

第1层 GAT:      16  → 每个头 32 → 4头拼接 → [N, 128]
                ReLU + Dropout

第2层 GAT:      128 → 每个头 32 → 4头拼接 → [N, 128]
                ReLU + Dropout

第3层 GAT:      128 → 每个头 32 → 4头拼接 → [N, 128]
                ReLU + Dropout

Linear:         128 → 32 → Dropout → 5

log_softmax → 输出预测 [N, 5]
```

### forward 流程

```python
def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch

    for i in range(self.num_layers):
        x = self.convs[i](x, edge_index)  # 消息传递（GraphSage 或 GAT）
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout)

    x = self.post_mp(x)                   # 分类头
    return F.log_softmax(x, dim=1)
```

------

## PyG 中的关键约定

### 命名约定

- `_j` 后缀 = **源节点**（邻居），如 `x_j`、`alpha_j`
- `_i` 后缀 = **目标节点**（中心节点），如 `x_i`、`alpha_i`
- `index` = 每条边对应的目标节点编号，用于 scatter 分组

### propagate 的工作流程

```
propagate(edge_index, x=(x_src, x_tgt), ...)
    → 按 edge_index 取出每条边的源/目标特征
    → 调用 message(x_j, ...)         # 对每条边
    → 调用 aggregate(inputs, index)   # 按目标节点分组
    → 调用 update(aggr_out)           # 可选后处理
    → 返回结果
```

### x=(x, x) 的含义

表示源节点和目标节点使用同一组特征。PyG 支持二部图，因此接口允许传入两组不同特征。在普通图（同构图）上，源和目标是同一批节点，所以传 `(x, x)`。

# DeepSNAP 

## DeepSNAP Basics

### Setup

DeepSNAP 构建在 PyTorch Geometric（PyG）之上，提供了更高层次的图数据管理、划分和任务定义接口。

```python
import torch
import deepsnap
import numpy as np
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch_geometric.datasets import Planetoid, TUDataset
```

### DeepSNAP Graph

DeepSNAP 的核心数据结构是 `Graph` 对象，它将 PyG 的 `Data` 或 NetworkX 的 `Graph` 封装成统一的表示。

```python
from torch_geometric.datasets import Planetoid

# 从 PyG 数据集创建 DeepSNAP Graph
pyg_dataset = Planetoid(root='./data', name='Cora')
graph = Graph(pyg_dataset[0])
```

一个 `Graph` 对象包含以下关键属性：

| 属性                 | 含义                                    |
| -------------------- | --------------------------------------- |
| `graph.node_feature` | 节点特征矩阵 `[num_nodes, feature_dim]` |
| `graph.edge_index`   | 边索引 `[2, num_edges]`                 |
| `graph.node_label`   | 节点标签                                |
| `graph.num_nodes`    | 节点数量                                |
| `graph.num_edges`    | 边数量                                  |

```python
print(f"节点数: {graph.num_nodes}")
print(f"边数:   {graph.num_edges}")
print(f"节点特征维度: {graph.node_feature.shape}")
print(f"类别数: {graph.node_label.max().item() + 1}")
```

#### What is the number of classes and the number of features?

通过 `graph.node_label` 和 `graph.node_feature` 可以直接获取：

```python
num_classes = graph.node_label.max().item() + 1
num_features = graph.node_feature.shape[1]
```

### DeepSNAP Dataset

`GraphDataset` 是管理多张图的容器，支持对图进行过滤、变换和划分。

```python
# 从 TUDataset 加载多图数据集（图分类场景）
pyg_dataset = TUDataset(root='./data', name='ENZYMES')
graphs = [Graph(pyg_dataset[i]) for i in range(len(pyg_dataset))]
dataset = GraphDataset(graphs, task='graph')

print(f"图的数量: {len(dataset)}")
print(f"第一张图: {dataset[0]}")
```

`GraphDataset` 还支持任务类型的声明，不同任务类型会影响后续的划分逻辑：

- `task='node'`：节点级任务（节点分类）
- `task='edge'` / `task='link_pred'`：边级任务（链路预测）
- `task='graph'`：图级任务（图分类）

#### What is the label of the graph (index)?

```python
# 获取指定索引图的标签
graph_index = 0
label = dataset[graph_index].graph_label
print(f"图 {graph_index} 的标签: {label}")
```

#### What is the number of edges for the graph (index)?

```python
# 获取指定索引图的边数
num_edges = dataset[graph_index].num_edges
print(f"图 {graph_index} 的边数: {num_edges}")
```

------

## DeepSNAP Advanced

### Setup

```python
import torch
import deepsnap
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from torch_geometric.datasets import Planetoid

pyg_dataset = Planetoid(root='./data', name='Cora')
graph = Graph(pyg_dataset[0])
```

### 图数据划分的特殊性

在传统机器学习中，数据划分很直接——把样本随机分成训练集、验证集和测试集。但在图数据上，情况复杂得多，因为样本之间通过拓扑结构相互关联。

图数据的划分需要回答两个核心问题：

1. **划什么？** 节点（Node Split）还是边（Edge Split）？
2. **怎么划？** 归纳式（Inductive）还是转导式（Transductive）？

### Inductive Split（归纳式划分）

**核心思想：** 训练集、验证集、测试集分别是完全独立的子图，彼此之间没有共享节点或边。

```
原始图 G
├── G_train（子图1：独立的节点集 + 内部边）
├── G_val  （子图2：独立的节点集 + 内部边）
└── G_test （子图3：独立的节点集 + 内部边）
```

**类比：** 用三个不同城市的社交网络分别做训练、验证和测试。模型必须学到可迁移的通用规律，而不是记住特定图的结构。

**适用场景：** 当你需要模型泛化到全新的、从未见过的图时（如蛋白质分类、分子属性预测）。

```python
dataset = GraphDataset(
    [graph],
    task='node',
    edge_train_mode='all'
)
train_set, val_set, test_set = dataset.split(
    transductive=False,  # False = 归纳式
    split_ratio=[0.8, 0.1, 0.1]
)
```

**关键特征：**

- 三个子图的节点集合互不相交
- 跨子图的边会被丢弃
- 模型在训练时完全看不到验证/测试子图的任何信息

### Transductive Split（转导式划分）

**核心思想：** 只有一张大图，所有节点在训练、验证、测试阶段都可见。划分的对象是"标签"或"边"，而非节点本身。

```
一张完整的图 G（所有节点始终可见）
├── 训练标签：部分节点/边有标签
├── 验证标签：另一部分节点/边有标签
└── 测试标签：剩余节点/边有标签
```

**类比：** 你认识全校所有学生（节点），也知道他们的友谊关系（边），但只知道一部分学生的专业（标签）。你需要根据社交关系推断其余人的专业。

**适用场景：** 引文网络节点分类（如 Cora）、社交网络链路预测。

```python
dataset = GraphDataset(
    [graph],
    task='link_pred',
    edge_train_mode='disjoint',
    edge_message_ratio=0.8
)
train_set, val_set, test_set = dataset.split(
    transductive=True,  # True = 转导式
    split_ratio=[0.8, 0.1, 0.1]
)
```

### Edge Level Split：链路预测的数据划分

链路预测是图机器学习的核心任务之一：给定一张图，预测两个节点之间是否存在（或将会存在）边。

#### All Mode（基础模式）

在 `edge_train_mode="all"` 模式下，训练阶段的所有训练边既用于 GNN 的消息传递（计算节点 Embedding），也用于计算损失函数。

```
训练阶段：
  消息传递边 = E_train（全部训练边）
  监督边     = E_train（全部训练边）← 和消息传递边完全相同！
```

**问题：数据泄露。** 模型在消息传递时通过边 (A, B) 将 B 的信息聚合到 A，然后转头就被问"A 和 B 之间有边吗？"——它当然知道有，因为刚才就是沿着这条边聚合的。这导致训练表现虚高，泛化能力极差。

#### Disjoint Mode（不相交模式）⭐

这是转导式链路预测的标准做法，也是 DeepSNAP 最精妙的设计。

**核心机制：** 在训练阶段，将训练边进一步拆分为两组互不相交的集合：

| 集合  | 名称                               | 用途                                  |
| ----- | ---------------------------------- | ------------------------------------- |
| E_mp  | 消息传递边 (Message Passing Edges) | 用于 GNN 卷积操作，计算节点 Embedding |
| E_sup | 监督边 (Supervision Edges)         | 作为正样本标签，用于计算 Loss         |

**关键约束：** E_mp ∩ E_sup = ∅（两者绝对不相交）

```
训练阶段（Disjoint 模式）：
  消息传递边 = E_train 的 80%（由 edge_message_ratio 控制）
  监督边     = E_train 的 20%（模型在计算 Embedding 时"看不见"这些边）

验证阶段：
  消息传递边 = E_train（全部训练边）
  监督边     = E_val

测试阶段：
  消息传递边 = E_train + E_val
  监督边     = E_test
```

**直觉理解——"破案游戏"类比：**

1. 我告诉你张三和李四是朋友，王五和赵六是亲戚（消息传递边）。
2. 我把陈七和周八之间的关系线藏起来，然后问你："陈七和周八是否有联系？"（监督边）。
3. 如果你能推理出正确答案，说明你真正学会了从拓扑结构中提取规律。

| 模式          | 做法                             | 后果                             |
| ------------- | -------------------------------- | -------------------------------- |
| All 模式      | 消息传递和监督用同一批边         | 模型直接"偷看"答案，过拟合严重   |
| Disjoint 模式 | 消息传递用 A 组边，监督用 B 组边 | 真正模拟"根据已知推断未知"的过程 |

实现参考：

```python
def disjoint_split(edge_index, edge_message_ratio=0.8):
    """
    将训练边划分为消息传递边和监督边。

    Args:
        edge_index: 边索引 [2, num_edges]
        edge_message_ratio: 消息传递边占比

    Returns:
        mp_edge_index:  消息传递边索引
        sup_edge_index: 监督边索引
    """
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)
    split = int(num_edges * edge_message_ratio)

    mp_edge_index = edge_index[:, perm[:split]]
    sup_edge_index = edge_index[:, perm[split:]]

    return mp_edge_index, sup_edge_index
```

[示意](.\disjoint_mode_step_by_step.html)

##### Resample Negative Edges（负样本重采样）

链路预测是二分类问题：正样本是图中存在的边，负样本是图中不存在的边。

**负采样逻辑：** 对于每条正样本边 (u, v)，随机采样一条不存在的边 (u, w) 作为负样本。

```python
def resample_negative_edges(graph, num_neg=1):
    """
    为每条正样本边生成对应的负样本边。
    确保采样到的负边不在原图中。
    """
    pos_edges = graph.edge_label_index[:, graph.edge_label == 1]
    neg_edges = []
    for i in range(pos_edges.shape[1]):
        src = pos_edges[0, i]
        while True:
            dst = torch.randint(0, graph.num_nodes, (1,))
            if not graph.has_edge(src, dst):
                neg_edges.append([src.item(), dst.item()])
                break
    return torch.tensor(neg_edges).t()
```

**为什么要每个 epoch 重新采样？** 用不同的负样本可以防止模型仅仅记住特定的负边模式，增强泛化能力。DeepSNAP 在每个 epoch 会自动执行重采样。

#### Graph Transformation and Feature Computation

在将图送入 GNN 之前，通常需要为节点和边计算初始特征。DeepSNAP 支持通过自定义变换函数完成：

```python
def transform_func(graph):
    """
    自定义图变换函数，为图添加/修改特征。
    """
    # 例：如果节点没有特征，使用节点度数作为初始特征
    if graph.node_feature is None:
        import networkx as nx
        G = graph.G  # 获取底层 NetworkX 图
        degrees = dict(G.degree())
        deg_tensor = torch.tensor(
            [degrees[i] for i in range(G.number_of_nodes())],
            dtype=torch.float
        ).unsqueeze(1)
        graph.node_feature = deg_tensor

    return graph

dataset = GraphDataset(
    [graph],
    task='link_pred',
    edge_train_mode='disjoint',
    edge_message_ratio=0.8,
    transform=transform_func
)
```

**常见变换操作：**

- **节点特征初始化：** one-hot 编码、度数、PageRank、聚类系数等拓扑指标
- **边特征计算：** 两端节点特征的拼接、差值、余弦相似度等
- **特征归一化：** 对连续特征做标准化或归一化处理

------

## Edge Level Prediction

将上述所有组件整合，构建完整的链路预测模型。

### 模型架构

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class LinkPredModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        """
        x:          节点特征矩阵 [num_nodes, input_dim]
        edge_index: 消息传递边索引 [2, num_mp_edges]
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def predict(self, z, edge_label_index):
        """
        z:                节点 Embedding [num_nodes, hidden_dim]
        edge_label_index: 监督边索引（包含正样本和负样本）
        """
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)  # 点积作为边存在概率
```

### 训练与评估

```python
model = LinkPredModel(input_dim=dataset.num_node_features, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCEWithLogitsLoss()

def train(train_graph):
    model.train()
    optimizer.zero_grad()

    # 用消息传递边计算节点 Embedding
    z = model(train_graph.node_feature, train_graph.edge_index)

    # 用监督边计算 Loss
    pred = model.predict(z, train_graph.edge_label_index)
    loss = criterion(pred, train_graph.edge_label.float())

    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(graph):
    model.eval()
    with torch.no_grad():
        z = model(graph.node_feature, graph.edge_index)
        pred = model.predict(z, graph.edge_label_index)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(
            graph.edge_label.cpu().numpy(),
            pred.sigmoid().cpu().numpy()
        )
    return auc

# 训练循环
for epoch in range(1, 201):
    loss = train(train_set[0])
    val_auc = evaluate(val_set[0])
    if epoch % 20 == 0:
        test_auc = evaluate(test_set[0])
        print(f'Epoch {epoch:03d} | Loss: {loss:.4f} | '
              f'Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}')
```

####  What is the maximum ROC-AUC score?

追求最大 ROC-AUC 时的调参方向：

- **隐藏层维度：** 64 / 128 / 256
- **GNN 层数：** 2-3 层（过深可能 Over-smoothing）
- **学习率：** 0.01 起步，配合 ReduceLROnPlateau
- **Dropout：** 0.3 - 0.5
- **边解码器：** 点积 → MLP 解码器 → 双线性解码器
- **edge_message_ratio：** 0.6 - 0.9 之间调整
- **负采样比例：** 调整正负样本的比例

