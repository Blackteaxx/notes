## 前置知识

随机向量$\mathbf{X} = (X_1, X_2, \cdots, X_n)$其实就是将$n$个随机变量组合在一起，形成一个向量。这个向量的分布称为随机向量的分布。随机向量的分布可以通过联合分布函数、联合概率密度函数、联合概率质量函数来描述。

期望：

$$
\mathbb{E}[\mathbf{X}] = \begin{pmatrix}
\mathbb{E}[X_1] \\
\mathbb{E}[X_2] \\
\vdots \\
\mathbb{E}[X_n]
\end{pmatrix}
$$

具有线性性质：

$$
\mathbb{E}[A\mathbf{X}] = A\mathbb{E}[\mathbf{X}]
$$

协方差矩阵：

回忆协方差的定义：$\text{cov}(X_i,X_j) = \mathbb{E}[(X_i - \mathbb{E}[X_i])(X_j - \mathbb{E}[X_j])]$

依旧是将$n$个随机变量组合在一起，形成一个向量。协方差矩阵是一个$n \times n$的矩阵，其中第$i$行第$j$列的元素是$X_i$和$X_j$的协方差。

$$
\text{cov}(\mathbf{X},\mathbf{X}) = \begin{pmatrix}
\text{cov}(X_1,X_1) & \text{cov}(X_1,X_2) & \cdots & \text{cov}(X_1,X_n) \\
\text{cov}(X_2,X_1) & \text{cov}(X_2,X_2) & \cdots & \text{cov}(X_2,X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{cov}(X_n,X_1) & \text{cov}(X_n,X_2) & \cdots & \text{cov}(X_n,X_n)
\end{pmatrix} \\
= \mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^T]
$$

由于协方差矩阵也是通过期望定义，因此也有线性性质。

同时，协方差矩阵是对称矩阵，且半正定，即对任意向量$\mathbf{a}$，有$\mathbf{a}^T \text{cov}(\mathbf{X},\mathbf{X}) \mathbf{a} \geq 0$

证明：

$$
\mathbf{a}^T \text{cov}(\mathbf{X},\mathbf{X}) \mathbf{a} = \mathbf{a}^T \mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^T] \mathbf{a} \\
= \mathbb{E}[\mathbf{a}^T (\mathbf{X} - \mathbb{E}[\mathbf{X}]) (\mathbf{X} - \mathbb{E}[\mathbf{X}])^T \mathbf{a}] = \mathbb{E}[(\mathbf{a}^T (\mathbf{X} - \mathbb{E}[\mathbf{X}]))^2] \geq 0
$$

对于协方差矩阵的极大似然估计为，是**有偏估计**：

$$
\hat{\Sigma} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{\mu})(x_i - \hat{\mu})^T
$$

无偏估计与一元情况类似，是：

$$
\hat{\Sigma} = \frac{1}{N-1} \sum_{i=1}^N (x_i - \hat{\mu})(x_i - \hat{\mu})^T
$$

其中$\hat{\mu}$是均值的极大似然估计。

## 问题描述

在很多书（例如 HDP），在高维世界，会发生维度灾难现象。

因此，我们需要对有结构的数据进行降维，有结构说明我们能通过某种数学映射将数据映射到低维空间。也即数据是一个低维空间的嵌入。

## MDS

MDS（Multi-Dimensional Scaling）是一种降维方法，它的目标是**保持高维空间中的距离关系**。MDS 的基本思想是，如果两个点在高维空间中距离很近，那么在低维空间中也应该距离很近。

我们原始数据为$X \in \mathbb{R}^{N \times D}$，我们希望将其映射到$Y \in \mathbb{R}^{N \times d}$，其中$d < D$。

那么我们定义原始数据的内积矩阵为$A = XX^T$。

我们对$A$进行特征值分解，得到$A = U \Lambda U^T$，其中$U$是特征向量矩阵，$\Lambda$是特征值矩阵。

也可以写成 rank 1 的形式：$A = \sum_{i=1}^D \lambda_i u_i u_i^T$。

由于我们要进行降维，因此我们只取非零/前$d$大特征值和特征向量，即$A' = \sum_{i=1}^d \lambda_i u_i u_i^T = Y Y^T$。

于是我们可以得到$Y = U_d \Lambda_d^{1/2}$，即为降维之后的数据向量。

## PCA

我们希望用正交空间中的一个超平面/子空间来近似数据，我们有两种方法：

- 最小化重构误差
- 最大化投影方差

### 最小化重构误差

我们假设超平面的一组正交基向量为$W = (w_1, w_2, \cdots, w_d)$，而空间中的一个点为$x$，可以使用基（扩展后）的线性组合来表示：$x = \sum_{i=1}^D \alpha_i w_i$，那么前$d$个线性组合系数为$\alpha_j = w^T_j x$，于是重构后坐标为$W^T x$, 重构后向量为 $W W^T x$, 重构误差为$\| x - W W^T x \|^2$。

于是我们可以得到总的重构误差为：

$$
\sum_{i=1}^N \| x_i - W W^T x_i \|^2 = \sum_{i=1}^N \| x_i \|^2 - \sum_{i=1}^N \| W^T x_i \|^2 \propto - \sum_{i=1}^N x_i^T W W^T x_i \\
= - \text{tr}(W^T X X^T W)
$$

其中$X = (x_1, x_2, \cdots, x_N)$, 我们可以得到优化目标为：

$$
\max_W \text{tr}(W^T X X^T W) \\
s.t. W^T W = I
$$

### 最大化投影方差

我们希望投影后的方差最大，即$\text{var}(W^T x)$最大，即$\text{var}(W^T x) = W^T \text{cov}(x) W$最大。

我们可以得到优化目标为：

$$
\max_W \text{tr}(W^T \text{cov}(x) W) \\
s.t. W^T W = I
$$

那么要求解这个问题，我们可以使用拉格朗日乘子法，得到：

$$
\text{cov}(x) W = W \Lambda
$$

其中$\Lambda$是特征值矩阵，$W$是特征向量矩阵。

因此我们要求解的$W$就是$\Lambda$的前$d$个特征向量。

而我们的投影后的数据就是$Y = W^T X$，投影后的各维度方差为$\text{var}(Y) = \text{diag}(W^T \text{cov}(x) W) = \text{diag} \Lambda$。

如果我们对数据进行中心化，那么我们的协方差矩阵就是$X^T X$，而我们的特征值矩阵就是$X^T X$的特征值矩阵。

而如果要**对降维后数据进行白化**，即只要让降维后的数据的协方差矩阵为单位矩阵，那么我们只需要对降维后的数据进行缩放即可，即$Y = \Lambda^{-1/2} W^T X$。

### SVD

实际上，PCA 的流程就是对中心化后的数据矩阵$\hat{X}$，的协方差矩阵$\hat{X} \hat{X}^T$进行特征值分解，得到特征值矩阵$\Lambda$和特征向量矩阵$W$。

其实我们仅要对数据矩阵进行奇异值分解，得到$\hat{X}= U \Sigma V^T$，那么我们可以得到：$W = U, \Lambda = \Sigma^2$

```{python}
face_matrix = np.array([face.flatten() for face in faces.values()])
face_matrix

k = 16

# 1. Compute the mean face

mean_face = face_matrix.mean(axis=0)

# 2. centerlized

center_face_matrix = face_matrix - mean_face

# 3. SVD

_, S, V = np.linalg.svd(center_face_matrix)

# 4. get max k S

idx = np.argsort(S)[-1:-k-1:-1]

eigenfaces = V[idx]

eigenfaces
```
