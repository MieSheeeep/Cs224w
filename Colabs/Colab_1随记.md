# PageRank 算法详解

 [pagerank-哔哩哔哩_bilibili](https://search.bilibili.com/all?vt=76481329&keyword=pagerank&from_source=webtop_search&spm_id_from=333.1007&search_source=5) 

## 1. 背景与直觉

PageRank 由 Google 创始人 Larry Page 和 Sergey Brin 提出，用于衡量网页的“重要性”。

**核心直觉：** 一个网页越重要，就会有越多的重要网页链接到它。

可以用一个类比来理解：学术论文被引用得越多，说明越有影响力；而如果引用它的论文本身也很有影响力，那它就更重要。PageRank 的逻辑完全一致。



---

## 2. 随机游走模型

PageRank 的数学本质是一个 **随机游走（Random Walk）** 过程：

1. 想象一个用户从某个网页出发。
2. 每一步，他有两种行为：
    * 以概率 $\beta$（阻尼系数，通常取 0.85）：点击当前页面上的某个链接，**均匀随机**地选一个。
    * 以概率 $1 - \beta$：不点链接，而是在浏览器地址栏随机输入一个网址（跳转到任意页面）。
3. 经过无限长时间后，用户在 **每个页面上停留的概率**，就是该页面的 PageRank 值。

---

## 3. 数学公式

### 3.1 基本更新公式

对于节点 $j$，每一轮迭代的更新公式为：

$$r_{new}(j) = \beta \times \sum_{i \to j} \frac{r(i)}{d_{out}(i)} + (1 - \beta) \times \frac{1}{N}$$

| 符号 | 含义 |
| :--- | :--- |
| $r(j)$ | 节点 $j$ 的 PageRank 值 |
| $\beta$ | 阻尼系数（damping factor），通常取 0.85 |
| $i \to j$ | 所有 **有边指向 $j$** 的节点 $i$ |
| $r(i)$ | 节点 $i$ 当前的 PageRank 值 |
| $d_{out}(i)$ | 节点 $i$ 的 **出度**（它指向多少个页面） |
| $N$ | 图中节点总数 |

### 3.2 两项的含义

**第一项：链接贡献项** $\beta \times \sum_{i \to j} \frac{r(i)}{d_{out}(i)}$
* 每个指向 $j$ 的节点 $i$，把自己的 PageRank 值 $r(i)$ **平均分配**给它的所有出边。
* $j$ 从每个这样的 $i$ 那里获得 $r(i) / d_{out}(i)$ 的贡献。
* 求和后乘以 $\beta$。

> **例子**：节点 A 的 PageRank 为 0.4，A 有 4 条出边，其中一条指向 B。那么 A 对 B 的贡献 = $0.4 / 4 = 0.1$。

**第二项：随机跳转项** $(1 - \beta) \times (1 / N)$
* 用户有 $1 - \beta$ 的概率随机跳转到任意页面。
* 跳转到每个页面的概率相等，都是 $1/N$。
* **这一项保证：**
    * 没有入边的节点也有非零的 PageRank。
    * 算法一定能收敛（不会在某些结构中陷入死循环）。

### 3.3 矩阵形式

将公式写成矩阵形式：

$$\mathbf{r} = \beta \times \mathbf{M} \times \mathbf{r} + (1 - \beta) \times \frac{1}{N} \times \mathbf{e}$$

其中：
* $\mathbf{r}$ 是 PageRank 向量 ($N \times 1$)。
* $\mathbf{M}$ 是转移矩阵 ($N \times N$)，$M_{ji} = 1 / d_{out}(i)$（如果 $i \to j$），否则为 $0$。
* $\mathbf{e}$ 是全 1 向量 ($N \times 1$)。

---

## 4. 迭代求解过程

PageRank 通过 **幂迭代法（Power Iteration）** 求解：

1.  **步骤 1：初始化**
    对所有节点 $j$，令 $r(j) = 1/N$。
2.  **步骤 2：迭代**
    重复以下过程：
    对每个节点 $j$：
    $$r_{new}(j) = \beta \times \sum_{i \to j} \frac{r(i)}{d_{out}(i)} + (1 - \beta) \times \frac{1}{N}$$
    令 $r = r_{new}$
    直到 $|r_{new} - r_{old}| < \epsilon$ （收敛阈值，如 $10^{-6}$）。
3.  **步骤 3：输出**
    输出 $r$ 作为最终 PageRank。

### 迭代示例
假设有 3 个节点的图：$A \to B \to C \to A$，$\beta = 0.8$。

* **初始状态**：$r(A) = r(B) = r(C) = 1/3 \approx 0.333$
* **第 1 轮迭代**：
    * $r(A) = 0.8 \times [r(C)/d_{out}(C)] + 0.2 \times (1/3) = 0.8 \times [0.333/1] + 0.067 = 0.333$
    * $r(B) = 0.8 \times [r(A)/d_{out}(A)] + 0.2 \times (1/3) = 0.8 \times [0.333/1] + 0.067 = 0.333$
    * $r(C) = 0.8 \times [r(B)/d_{out}(B)] + 0.2 \times (1/3) = 0.8 \times [0.333/1] + 0.067 = 0.333$
    *(对称图，所以每个节点的 PageRank 相等，一轮就收敛了)*

---

## 5. 特殊问题与处理

### 5.1 Dead Ends（死胡同）
有些节点没有出边（比如一个没有任何外链的网页）。
* **问题**：随机游走走到这里就“卡住”了，PageRank 值会泄漏到 0。
* **解决**：从死胡同节点以均匀概率跳转到任意节点（等价于给它添加指向所有节点的边）。

### 5.2 Spider Traps（蜘蛛陷阱）
一组节点形成一个封闭的环，只有内部链接，没有出边指向外部。
* **问题**：随机游走一旦进入就出不来，这组节点会吸收所有的 PageRank。
* **解决**：阻尼系数 $\beta < 1$ 就是为了解决这个问题。$1 - \beta$ 的随机跳转概率保证用户能“逃出”陷阱。



---

## 6. 有向图 vs 无向图

| 维度 | 有向图 | 无向图 |
| :--- | :--- | :--- |
| **边的含义** | $A \to B$ 表示 A 链接到 B | $A-B$ 表示双向连接 |
| **出度** | 只计算出边数量 | 度 = 邻居数量（既是入度也是出度） |
| **典型场景** | 网页链接、论文引用 | 社交网络中的好友关系 |
| **公式中** | 区分 $i \to j$ 的方向 | 邻居即是“指向 $j$”的节点 |

在无向图中，每条边等价于两条方向相反的有向边，所以邻居既“指向你”，你也“指向邻居”。

---

## 7. 代码对应关系

```python
def one_iter_pagerank(G, beta, r0, node_id):
    r1 = 0
    N = G.number_of_nodes()                    # 节点总数

    for neighbor in G.neighbors(node_id):       # 遍历所有指向 node_id 的节点
        di = G.degree(neighbor)                 # 邻居的出度 d_out(i)
        r1 += beta * (r0 / di)                 # 累加 beta * r(i) / d_out(i)

    r1 += (1 - beta) * (1 / N)                 # 加上随机跳转项
    r1 = round(r1, 2)
    return r1
```

**注意：** 这里 $r_0 = 1/N$ 对所有节点相同，是因为只做了第一轮迭代（初始值均匀分配）。如果做多轮迭代，每个节点的 $r$ 值会不同，需要用一个字典来存储。

---

## 8. 关键性质总结

1.  **所有节点的 PageRank 值之和 = 1**（概率分布）。
2.  **阻尼系数 $\beta$ 越大**，链接结构的影响越大；$\beta$ 越小，越接近均匀分布。
3.  **收敛性有保证**：只要 $\beta < 1$，幂迭代法一定收敛。
4.  **入边多 $\neq$ 一定重要**：关键是入边来源的质量（重要节点的链接更有价值）。
5.  **出度高的节点**给每个邻居的贡献更少（重要度被稀释）。



# PCA



 [12. 主成分分析：看见数据的主方向 | 数学不难：线性代数_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1DT6RBqEyW?spm_id_from=333.788.videopod.sections&vd_source=4c201d998e0456ea07448438d582be04) 