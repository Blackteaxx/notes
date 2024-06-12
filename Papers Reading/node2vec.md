## 参考

[KDD 2016 | node2vec：Scalable Feature Learning for Networks](https://blog.csdn.net/Cyril_KI/article/details/121644349)

## Introduction

在网络的分析之中，特征工程与时间复杂度是我们需要着重考虑的事情，因为网络的规模往往是非常大的，因此我们需要一种能够快速提取网络（节点）特征的方法。

而以往基于 Matrix Factorization 的方法，往往需要将网络转化为邻接矩阵，然后再进行矩阵分解，**这样的方法在网络规模较大的时候，往往会导致时间复杂度过高**。

本文的主要贡献是定义了一个灵活的**节点网络邻域概念**。通过一个有偏随机游走来有效地探索给定节点的不同社区，然后返回一个游走序列。

算法是灵活的，可以通过可调参数控制搜索空间，而不是像之前的方法那样死板。

搜索什么？我们需要**学到一个节点的特征向量表示**，在此之前，我们首先要找到该节点的邻域节点，这些节点可以通过某种方法来生成，而这也是本文的重点。

## Related Work

以往的基于图的 Representation Learning 方法，往往是基于**邻接矩阵**/**拉普拉斯矩阵（Degree - Adjancy ）**的，比如 DeepWalk、LINE 等，这些方法的时间复杂度往往是$O(n^2)$，因此在大规模网络上的应用会受到限制。

word2vec 使用 N 元语法来学习词向量。

## Feature Learning

我们希望学到一个节点的特征向量表示$f:\mathbb{R}^N \to \mathbb{R}^d$, 而我们通过采样获取每一个节点的邻域节点$N_s(u)$，将 skip-gram 模型应用到这个邻域节点上，来学习节点的特征向量，我们最大化如下的目标函数：

$$
\max \prod_{u \in V} p(N_s(u)|f) \iff\max_f \sum_{u \in V} \log p(N_s(u)|f) \iff \max_f \sum_{u \in V} \sum_{v \in N_s(u)} \log p(v|f(u))
$$

在邻域出现独立的假设下，我们增加特征空间对称性的假设（不是很理解这一块怎么来的），那么我们可以使用 softmax 函数来表示$p(v|f(u))$：

$$
p(v|f(u)) = \frac{\exp(f(v) \cdot f(u))}{\sum_{v' \in V} \exp(f(v') \cdot f(u))}
$$

那么目标函数就变为：

$$
\max_f \sum_{u \in V} \sum_{v \in N_s(u)} \log \frac{\exp(f(v) \cdot f(u))}{\sum_{v' \in V} \exp(f(v') \cdot f(u))} \iff \\
\max_f \sum_{u \in V} \sum_{v \in N_s(u)} f(v) \cdot f(u) - \log \sum_{v' \in V} \exp(f(v') \cdot f(u)) \iff \\
\max_f \sum_{u \in V} - \log \sum_{v' \in V} \exp(f(v') \cdot f(u)) + \sum_{v \in N_s(u)} f(v) \cdot f(u)
$$

大型网络下，计算$\sum_{v' \in V} \exp(f(v') \cdot f(u))$是非常耗时的，因此我们需要一种更快的方法来计算这个值（负采样）。

## Neighbor Sampling

给定一个源节点$u$，我们的任务是找到它的邻域节点集合 $N_s(u)$

根据**同质性假设**，属于相似网络集群的高度互联节点应该嵌入到一起

而根据**结构等价假设**，在网络中具有相似结构角色的节点应该嵌入到一起，虽然属于不同网络集群，但是它们都是相应集群的中枢节点，角色类似。

**结构等价不同于同质性**，它不强调节点间必须互连，它们可能距离很远，这一点很重要。

BFS 仅限于与源节点相连的节点，倾向于在初始节点的周围游走，可以反映出一个节点的邻居的微观特性。

而 DFS（可以重复访问）一般会跑得离源节点越来越远，可以反映出一个节点邻居的宏观特性。

两种方法的局限都很明显：BFS 只能探索图的很小一部分，而 DFS 移动到更大的深度会导致复杂的依赖关系，因为采样节点可能远离源，并且可能不太具有代表性。

于是就提出了Node2Vec

## Node2Vec

![img](https://img2023.cnblogs.com/blog/3436855/202405/3436855-20240529233936674-1845228202.png)