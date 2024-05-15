## 1. 习题一(multi-dimension PCA)

![img](https://img2023.cnblogs.com/blog/3436855/202405/3436855-20240515194454455-2039918970.png)

符号表示：

$$
\text{Cov}(x) = \frac{1}{N} \sum (x_i - \bar{x})(x_i - \bar{x})^T
$$

在一维 PCA 降维中，我们有：

$$
x_i \approx \bar{x} + \xi_1^T (x_i - \bar{x})\xi_1
$$

### (a)

令$y_i = x_i - \xi_1^T (x_i - \bar{x})\xi_1$, 已知$\text{Cov}(x) = \sum \lambda_i \xi_i \xi_i^T$，则有：

$\bar{y}$可以表示为

$$
\bar{y} = \frac{1}{N} \sum (x_i - \xi_1^T (x_i - \bar{x})\xi_1) = \bar{x} - \xi_1^T (\bar{x} - \bar{x})\xi_1 = \bar{x}
$$

同时由 ONB 特征向量定义，可得

$$
x_i - \bar{x} = \sum_{j=1}^D \xi_j^T (x_i - \bar{x})\xi_j
$$

于是，有

$$
\text{Cov}(y) = \frac{1}{N} \sum_{i=1}^N (\sum_{j=2}^D \xi_j^T (x_i - \bar{x})\xi_j)(\sum_{j=2}^D \xi_j^T (x_i - \bar{x})\xi_j)^T
$$

其中内部求和部分为

$$
\frac{1}{N} \sum_{i=1}^N  \sum_{j=2}^D \sum_{k=2}^D (\xi_j^T (x_i - \bar{x})) (\xi_k^T (x_i - \bar{x})) \xi_j \xi_k^T \\
=  \sum_{j=2}^D \sum_{k=2}^D \xi_j \xi_j^T  \frac{1}{N} \sum_{i=1}^N  (x_i - \bar{x}) (x_i - \bar{x})^T \xi_k^T \xi_k^T\\=
\sum_{j=2}^D \sum_{k=2}^D \xi_j \xi_j^T \text{Cov}(x) \xi_k \xi_k^T \\=
\sum_{j=2}^D \sum_{k=2}^D \xi_j \xi_j^T \lambda_k \xi_k \xi_k^T = \sum_{j=2}^D \lambda_j \xi_j \xi_j^T
$$

因此，在多维 PCA 降维中，$\text{Cov}(y) = \sum_{j=2}^D \xi_j \xi_j^T$，即在降$p$维后，剩下的数据继续降维会沿着剩余特征值最大的方向。

### (b)

验证(a)中的结论

## 2 习题二(瑞利商)

![img](https://img2023.cnblogs.com/blog/3436855/202405/3436855-20240515194534168-640155064.png)

给定$S_B$与$S_W$为两个$n\times n$的实对称矩阵，若存在$\lambda$使得$S_Bw= \lambda S_W w$则称$\lambda$为广义特征值。

### (a)

求广义特征向量之间带权正交，即证明$i=j$时，$w_i^TS_Ww_j = 1$，否则为 0

对于 cholesky 分解，假设$S_W$是`real symmetric positive definite matrix`,有$S_W = LL^T$，则有

$$
S_Bw= \lambda LL^T w
$$

其中 L 为满秩矩阵，因此有

$$
L^{-1}S_BL^{-T}L^Tw = \lambda L^Tw
$$

由于$L^{-1}SL^{-T}$是对称矩阵，根据对称矩阵的相似对角化，因此有$L^Tw$是一组 ONB，其中

$$
w^T L L^T w = w^T S_W w = 1
$$

### (b)

求广义瑞利商$J(w) = \frac{w^T S_B w}{w^T S_W w}$的最大值

沿用上一问的假设与结论，有$w = \sum_i^D \alpha_i w_i$，其中的$w_i$ 为 orthogonormal basis, 于是有

$$
J(w) = \frac{w^T S_B w}{w^T S_W w} = \frac{\sum_i^D \alpha_i^2 \lambda_i}{\sum_i^D \alpha_i^2}
$$

优化问题可以定义为

$$
\sum_i^D \alpha_i^2 \lambda_i \to \max/\min, \\\quad s.t. \sum_i^D \alpha_i^2 = 1
$$

于是可知，$\max J(w) = \lambda_1$, $\min J(w) = \lambda_n$

### (c)

求解行列式的值

$$
J = \frac{|W^T\Lambda S_W W|}{|I|} = \prod_i^d \lambda_i
$$

## 3 习题三(核函数)

## 4 习题四(SVM)

![img](https://img2023.cnblogs.com/blog/3436855/202405/3436855-20240515204416390-1489249355.png)

### (a)

$$
\arg \min_{w,b} \frac{1}{2} w^Tw + C \sum_{i=1}^N \xi_i \mathbb{I}(y_i = 1) + kC \sum_{i=1}^N \xi_i \mathbb{I}(y_i = -1) \\
s.t. \quad y_i(w^Tx_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 , \quad \forall i = 1,2,\cdots, N
$$

### (b)

拉格朗日函数为：

$$
\mathcal{L}(w,b,\xi,\alpha,\beta) = \frac{1}{2}w^Tw + C \sum_{i=1}^N \xi_i \mathbb{I}(y_i = 1) + kC \sum_{i=1}^N \xi_i \mathbb{I}(y_i = -1) + \\
\sum_{i=1}^N \alpha_i(1 - \xi_i - y_i(w^Tx_i + b)) - \sum_{i=1}^N \beta_i \xi_i
$$

对偶问题为:

$$
\arg \max_{\alpha,\beta} \min_{w,b,\xi} \mathcal{L}(w,b,\xi,\alpha,\beta) \\
s.t. \quad \alpha_i \geq 0, \quad \beta_i \geq 0, \quad \forall i = 1,2,\cdots, N
$$

我们可以求解对偶问题，得到最优解$\alpha^*, \beta^*$，然后求解原问题的最优解。

我们分别对$w,b,\xi$求导，得到

$$
\frac{\partial}{\partial w} \mathcal{L} = w - \sum_{i=1}^N \alpha_i y_i x_i = 0 \Rightarrow w = \sum_{i=1}^N \alpha_i y_i x_i \\
$$

$$
\frac{\partial}{\partial b} \mathcal{L} = -\sum_{i=1}^N \alpha_i y_i = 0 \Rightarrow \sum_{i=1}^N \alpha_i y_i = 0
$$

$$
\frac{\partial}{\partial \xi_i} \mathcal{L} = C\mathbb{I}(y_i = 1) + kC\mathbb{I}(y_i = -1) - \alpha_i - \beta_i = 0 \\ \Rightarrow \alpha_i + \beta_i = C\mathbb{I}(y_i = 1) + kC\mathbb{I}(y_i = -1)
$$

代入上述三个式子，化简$\mathcal{L}$

$$
\mathcal{L}(w,b,\xi,\alpha,\beta) = \frac{1}{2}w^Tw + C \sum_{i=1}^N \xi_i \mathbb{I}(y_i = 1) + kC \sum_{i=1}^N \xi_i \mathbb{I}(y_i = -1) + \\
\sum_{i=1}^N \alpha_i(1 - \xi_i - y_i(w^Tx_i + b)) - \sum_{i=1}^N \beta_i \xi_i \\
= \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_{i=1}^N \alpha_i (1-y_i(w^Tx_i + b))  \\ - \sum_{i=1}^N \alpha_i \xi_i - \sum_{i=1}^N \beta_i \xi_i + C \sum_{i=1}^N \xi_i \mathbb{I}(y_i = 1)+ kC \sum_{i=1}^N \xi_i\mathbb{I}(y_i = -1) \\
= \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_{i=1}^N \alpha_i - \sum_{i=1}^N \alpha_i y_i(w^Tx_i + b) \\
= \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j + \sum_{i=1}^N \alpha_i - \sum_{i=1}^N \alpha_i y_i\sum_{i=1}^N \alpha_i y_i x_i^Tx_i \\
= \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j
$$

那么对于$\beta_i$，我们可以得到

$$
\beta_i = C\mathbb{I}(y_i = 1) + kC\mathbb{I}(y_i = -1) - \alpha_i \geq 0 \\
\Rightarrow \alpha_i \leq C\mathbb{I}(y_i = 1) + kC\mathbb{I}(y_i = -1)
$$

因此对偶问题为：

$$
\arg \max_{\alpha} \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j \\
s.t. \quad 0 \leq \alpha_i \leq C\mathbb{I}(y_i = 1) + kC\mathbb{I}(y_i = -1), \quad \forall i = 1,2,\cdots, N \\
\sum_{i=1}^N \alpha_i y_i = 0, \quad \forall i = 1,2,\cdots, N
$$

KKT 条件为：

$$
\begin{cases}
\alpha_i \geq 0 \\
\beta_i \geq 0 \\
\alpha_i(1 - \xi_i - y_i(w^Tx_i + b)) = 0 \\
\beta_i \xi_i = 0 \\
y_i(w^Tx_i + b) \geq 1 - \xi_i \\
\xi_i \geq 0 \\
\end{cases}
$$

对于 w：

$$
w = \sum_{i=1}^N \alpha_i y_i x_i \\
$$

对于 b，我们发现有当$y_i(wx_i + b) = 1 - \xi_i$时，$\alpha_i(1 - \xi_i - y_i(w^Tx_i + b)) = 0$才会有$\alpha_i \neq 0$

$$
b = y_i - y_i\xi_i - w^Tx_i, \quad \forall i \text{ satisfy }y_i(wx_i + b) = 1 - \xi_i
$$

对于 $\xi_i$，我们有

$$
\xi_i = \max(0, 1 - y_i(w^Tx_i + b))
$$

## 5 习题五(朴素贝叶斯)

![img](https://img2023.cnblogs.com/blog/3436855/202405/3436855-20240515212838169-291145375.png)

### (a)

基本假设为特征变量在给定类别下条件独立性假设。
优点是带来了计算数据似然条件概率的简化，即不用对所有变量求解积分；能够解决数据稀疏问题。
局限性是在现实条件下，特征变量很难满足条件独立性假设。
