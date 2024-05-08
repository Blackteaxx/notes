## 1. 习题一(multi-dimension PCA)

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

给定$S_B$与$S_W$为两个$n\times n$的实对称矩阵，若存在$\lambda$使得$S_Bw= \lambda S_W w$则称$\lambda$为广义特征值。

### (a)

求广义特征向量之间带权正交，即证明$i=j$时，$w_i^TS_Ww_j = 1$，否则为 0

对于 cholesky 分解，假设$S_W$是`real symmetric positive definite matrix`,有$S_W = LL^T$，则有

$$
S_Bw= \lambda LL^T w
$$

其中 L 为满秩矩阵，因此有

$$
L^{-1}SL^{-T}L^Tw = \lambda L^Tw
$$

由于$L^{-1}SL^{-T}$是对称矩阵，因此有$L^Tw$是一组 ONB，其中

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
