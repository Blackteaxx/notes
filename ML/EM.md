# EM Algorithm

## 1. GMM 引入

若数据服从一个高斯分布，可以用 MLE + 偏导为 0 的方法求解参数，其中框架如下

$$
\theta = \{ \mu, \Sigma \} = \arg \max_{\theta} \sum_{i=1}^N \log p(x_i | \theta)
$$

而由于单个高斯分布概率密度函数为 convex，因此整体的似然函数也是 convex 的，梯度为 0 点即为最优解。

然而，若数据服从多个高斯分布，density function 为

$$
p(x) = \sum_{k=1}^K \alpha_k \mathcal{N}(x | \mu_k, \Sigma_k) \\
\quad s.t. \sum_{k=1}^K \alpha_k = 1
$$

若使用 MLE 方法，log likelihood 为

$$
LL = \sum_{i=1}^N \log p(x_i) = \sum_{i=1}^N \log \sum_{k=1}^K \alpha_k \mathcal{N}(x_i | \mu_k, \Sigma_k)
$$

由于 density function 不再是凸函数，同时等式直接计算复杂，因此无法直接求解最优解，此时可以引入 EM 算法，为一个迭代算法。

## 2. EM 算法收敛性

EM 算法的迭代过程如下

$$
\begin{align}
    \theta^{(t+1)} = \arg \max_{\theta} \text{E}_{p(z|x;\theta^{(t)})}[\text{log} p(x,z;\theta)] = \arg \max_{\theta} \int_z p(Z | X, \theta^{(t)}) \log p(X, Z | \theta)
\end{align}
$$

迭代算法欲证明收敛性，只需要证明每次迭代后的 log likelihood 均不减少即可,即：

$$
LL(\theta^{(t+1)}) \geq LL(\theta^{(t)}) \\
p(X|\theta^{(t+1)}) \geq p(X|\theta^{(t)}) \\
\text{log} p(X|\theta^{(t+1)}) \geq \text{log} p(X|\theta^{(t)})
$$

$proof$:

$$
\text{log} p(X|\theta) = \text{log} p(X,Z|\theta) - \text{log} p(Z|X,\theta) \\
$$

对两边求期望，有

$$
\text{E}_{p(Z|X,\theta^{(t)})}[\text{log} p(X|\theta)] = \text{E}_{p(Z|X,\theta^{(t)})}[\text{log} p(X,Z|\theta)] - \text{E}_{p(Z|X,\theta^{(t)})}[\text{log} p(Z|X,\theta)]
$$

由于左边与$Z$无关，因此有

$$
\text{log} p(X|\theta) = \text{E}_{p(Z|X,\theta^{(t)})}[\text{log} p(X,Z|\theta)] - \text{E}_{p(Z|X,\theta^{(t)})}[\text{log} p(Z|X,\theta)]
$$

右边展开为积分形式，有

$$
\text{log}  p(X|\theta) = \underbrace{\int p(Z|X,\theta^{(t)}) \text{log} p(X,Z|\theta) dz}_{Q(\theta,\theta^{(t)})} - \underbrace{\int p(Z|X,\theta^{(t)}) \text{log} p(Z|X,\theta) dz}_{H(\theta,\theta^{(t)})}
$$

由于 EM 算法的形式,可知最大化 $Q$项，因此有

$$
Q(\theta^{t+1},\theta^{(t)}) \geq Q(\theta,\theta^{(t)})
$$

即有

$$
Q(\theta^{(t+1)},\theta^{(t)})  \geq Q(\theta^{(t)},\theta^{(t)})
$$

那么若要证明 EM 算法的收敛性，只需要证明 $H$ 项的减少即可。

$$
H(\theta^{(t+1)},\theta^{(t)}) \leq H(\theta^{(t)},\theta^{(t)})
$$

即证

$$
H(\theta,\theta^{(t)}) \leq H(\theta^{(t)},\theta^{(t)}) \\
H(\theta^{(t)},\theta^{(t)}) - H(\theta,\theta^{(t)}) \geq 0
$$

展开即有

$$
\int p(Z|X,\theta^{(t)}) \text{log} \frac{p(Z|X,\theta^{(t)})}{p(Z|X,\theta)} dz = KL(p(Z|X,\theta^{(t)})\| p(Z|X,\theta)) \geq 0
$$

收敛性得证，但无法证明收敛到全局最优解。

## 3. EM 算法公式导出

对于 n 个数据$(x_1, \dots,x_N)$,MLE 如下：

$$
\theta = \arg \max_{\theta}\sum_N \log p(x_i|\theta)
$$

若带有隐变量$z=(z_1,\dots,z_k)$，则有

$$
\theta = \arg \max_{\theta}\sum_N \log p(x_i|\theta) = \arg \max_{\theta}\sum_N \log \sum_z p(x_i,z_j|\theta)
$$

由于直接求解困难，因此使用$z$的分布放缩:

$$
\sum_N \log \sum_z p(x_i,z|\theta) = \sum_N \log \sum_z q_i(z_j) \frac{p(x_i,z_j|\theta)}{q_i(z_j)} \geq \sum_N \sum_z q_i(z_j) \log \frac{p(x_i,z_j|\theta)}{q_i(z_j)}
$$

上述过程描述了$LL(\theta)$的下界，而我们希望$\arg \max LL(\theta)$，因此我们希望下界能取到等号，即我们希望等式成立，根据 Jensen 不等式，等式成立当且仅当$\frac{p(x_i,z_j|\theta)}{q(z_j)} = c$,$c$为常数，且$\sum q_i(z_j) = 1$，因此有

$$
q_i(z_j) = \frac{p(x_i,z_i,\theta)}{\sum p(x_i,z_i,\theta)} = p(z_j|x_i,\theta)
$$

- 因此在 E 步，我们需要求解$q_i(z_j)$的分布，尽可能使得$LL(\theta)$的下界取到最大值，
- 而在 M 步，我们获取了$q_i(z_j)$的分布之后，可以对$LL(\theta)$求导，得到最优解$\theta$，即为 EM 算法。

## 参考

[EM 算法详解](https://zhuanlan.zhihu.com/p/40991784)
[徐亦达机器学习：Expectation Maximization EM 算法 【2015 年版-全集】](https://www.bilibili.com/video/BV1Wp411R7am/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click)
