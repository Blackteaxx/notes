## Background

在概率图模型的推断过程中，当过程比较复杂时，需要进行大量的计算，这时候就需要使用一些近似推断的方法。

## Monte Carlo Method

一种基于采样的随机近似方法，主要用途是数值积分。

而我们经常通过一个概率$p(Z|X)$，其中$Z$是我们 latent variable，$X$是 observed variable，来求取期望$\mathbb{E}_{Z|X}[f(Z)] = \int f(Z) p(Z|X) \text{d}Z$，此时可以使用 Monte Carlo 通过采样求积分, $\int f(Z) p(Z|X) \text{d}Z \approx \frac{1}{N} \sum f(Z_i)$, 那么问题就转化为了如何从复杂分布中采样$Z_i$。

1. 概率分布采样

通过计算机能够产生一个均匀分布的随机数，然后通过给定的 pdf $p(z)$获得 cdf，然后通过 cdf 的反函数得到采样。

2. 拒绝采样 (Rejection Sampling)

先指定一个 proposal distribution $q(z)$，使得 $q(z) \geq M p(z)$，其中 $M$ 是一个常数，指定接受率$\alpha=\frac{p(z_i)}{Mq(z_i)}$, 算法描述为：

a. 从 $q(z)$ 中采样 $z_i$
b. 在 0-1 均匀分布中采样 $u$
c. 如果 $u \leq \alpha$ 则接受 $z_i$，否则拒绝

即，我们其实知道$p(z_i)$在某一点的取值，但是我们不知道整体的分布，因此我们通过一个简单的分布$q(z)$来近似，然后通过接受率来判断是否接受。

3.  重要性采样 (Importance Sampling)

$\mathbb{E}\_{ p(z) }[f(z)] = \int p(z) f(z) \text{d}z = \int \frac{p(z)}{q(z)} q(z) f(z) \text{d} z \approx \frac{1}{N}\sum f(z_i) \frac{p(z_i)}{q(z_i)} $

严重依赖于 q 的选择，如果 q 选择不当，会导致采样的效率很低。

4. Sampling Importance Resampling (SIR)

## Markov Chain

随机过程研究的是一个随机变量序列，而马尔可夫链是一种特殊的随机过程。

马尔可夫链是一个事件和状态都是离散的，具有齐次$n$阶马尔可夫性的随机过程，即给定现在的状态，未来的状态与过去的状态无关。
1 阶 Markov Chain：$p(x_t | x_{t-1}, x_{t-2}, \dots, x_1) = p(x_t | x_{t-1})$

我们定义转移矩阵$p_{ij} = p(x_{t+1} = j | x_t = i)$

以概率图模型表示，即为

```mermaid
graph LR
    x1 --> x2 --> x3 --> ... --> xt --> ....
```

$t$时刻状态的$x$取值可以由$t-1$时刻的状态+转移概率求边缘概率得到。

**平稳分布**：$\pi = \pi P$，其中$\pi$是一个行向量，$P$是转移矩阵，$\pi$是一个平稳分布，即$\pi = \pi P = \pi P^2 = \dots$，也可以表示为$\pi(i) = \sum_j \pi(j) p_{ji}$

**Detailed balance condition**：$\pi(i) p_{ij} = \pi(j) p_{ji}$

$\text{Detailed balance condition} \Rightarrow \text{Balance Distribution}$

> Proof

$$
\sum_j \pi(j) p_{ji} = \sum_j \pi(i) p_{ij} =  \pi(i) \sum_j p_{ij} = \pi(i)
$$

假如我们能够构造一个马尔可夫链，使得其平稳分布为我们要求的分布，那么我们就可以通过马尔可夫链的采样来获得我们要求的分布。

## MH Algorithm

主要思想：从一个 Markov Chain 中不断地采样，使得其平稳分布为我们要求的分布。

我们需要构造转移矩阵，使得其平稳分布为我们要求的分布。但是不能直接构造出这样的矩阵，但是通过detailed balance condition可以构造平稳分布，因此我们先构造一个提议分布$Q(x^\star | x^{t-1})$，然后通过构造一个接受率来使得满足以下条件：

$$
p(x^{t-1}) Q(x^\star | x^{t-1}) A(x^\star| x^{t-1}) = p(x^\star) Q(x^{t-1} | x^\star) A(x^{t-1} | x^\star)
$$

如果我们定义接受率为$A(x^\star| x^{t-1}) =\min(1, \frac{p(x^\star) Q(x^{t-1} | x^\star)}{p(x^{t-1}) Q(x^\star | x^{t-1})}$) ，那么上述等式就可以满足 detailed balance condition。

> Proof

那么上述等式左边即可被表示为：

$$
p(x^{t-1}) Q(x^\star | x^{t-1}) A(x^\star| x^{t-1}) = \\
p(x^{t-1}) Q(x^\star | x^{t-1}) \min(1, \frac{p(x^\star) Q(x^{t-1} | x^\star)}{p(x^{t-1}) Q(x^\star | x^{t-1})} )= \\
\min(p(x^{t-1}) Q(x^\star | x^{t-1}), p(x^\star) Q(x^{t-1} | x^\star)) = \\
p(x^{t-1}) Q(x^\star | x^{t-1}) \min(1, \frac{p(x^\star) Q(x^{t-1} | x^\star)}{p(x^{t-1}) Q(x^\star | x^{t-1})} ) = \\
p(x^\star) Q(x^{t-1} | x^\star) A(x^{t-1} | x^\star)
$$

即我们可以通过构造一个接受率为$A(x^\star| x^{t-1}) =\min(1, \frac{p(x^\star) Q(x^{t-1} | x^\star)}{p(x^{t-1}) Q(x^\star | x^{t-1})}$)的转移矩阵，使得其平稳分布为我们要求的分布。

> End Proof

那么根据拒绝采样的思想，我们可以通过一个简单的分布$q(x)$来近似$p(x)$，然后通过接受率来判断是否接受。

**这里的接受率在算法当中是怎么计算出来的？**
