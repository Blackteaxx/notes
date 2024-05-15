---
Title: Information Theory
---

## Entropy

### Topic 1: 定义在事件上的函数

#### 自信息

自信息是一个事件的信息量的度量，基本思想是概率越小，事件蕴含的信息量越大，满足如下性质：

1. 非负性：$f(x) \geq 0$
2. 单调：如果事件$a,b$, $P(a) < P(b)$, 则 $f(a) > f(b)$
3. $f(a) = 0$ iff $P(a) = 1$
4. $P(a) = 0$ 则 $f(a) = \infin$
5. 独立可加性：$f(a, b) = f(a) + f(b)$ when $a$ and $b$ are independent.

可以证明：$f(x) = -\log P(x)$ 满足上述性质。

定义：样本空间中的一个事件 $x$ 的自信息为

$$
I(x) = -\log P_X(x)
$$

单位为$bit$

Insight:

- 自信息是定义在一个事件上的，而不是一个分布上。
- 在发生前，自信息表示的是不确定性
- 在发生后，自信息表示的是信息量

#### 联合自信息

定义：样本空间中两个事件 $x, y$ 的概率的联合自信息为

$$
I(x,y) = -\log P_{XY}(x,y)
$$

#### 条件自信息

定义：样本空间中，给定事件 $y$ 发生的条件下，事件 $x$ 的条件自信息为

$$
I(x|y) = -\log P_{X|Y}(x|y)
$$

Insight:

- $y=b_i$给定时，$x$ 发生前的不确定性
- $y=b_i$给定时，$x$ 发生后的信息量

**自信息之间的联系**：

$$
I(x,y) = -\log P_{XY}(xy) = -\log P_{X|Y}(x|y) P_Y(y) = I(x|y) + I(y)
$$

同理

$$
I(x,y) =  I(y|x) + I(x)
$$

#### 互信息

已知 I(x)是 x 事件所含有的信息量，I(x|y)是 x 事件在 given y 事件发生后的信息量，那么可以定义两者的差值为 y 事件带给 x 事件的信息量（增益）

(此处添加下标，其实上文也应该添加下标，指给定样本空间的意思)

$$
I_{X;Y}(x;y) = I_X(x) - I_{X|Y}(x|y) = \log \frac{P(x|y)}{P(x)} = \log \frac{P(xy)}{P(x)P(y)}
$$

互信息的性质：

- $I(x;y) = I(y;x)$
- 当$x,y$独立时，I(x;y) = 0 ($y$无法给$x$带来信息)
- ==可正可负==
- $I(x;y) \leq I(x) / I(y)$

额外的条件互信息:

$$
I(x;y | z) = \log \frac{P(x | y,z)}{P(x | z)} = \log \frac{P(x, y, z)}{P(x|z)P(y|z)}
$$

### Topic 2:定义在概率分布上的函数

#### (离散)信息熵

定义为一个样本空间上所有随机事件（随机变量是离散的）的自信息的期望，熵在物理意义上是平均意义下对随机事件不确定性/信息量的度量，计算机意义上是**平均意义上对随机变量的编码长度**。

_Example：投掷均匀硬币的信息熵为 1bit，即可以使用一位编码表示所有结果_

$$
H(X) = E_X[I(X)] = - \sum_i^n p(x_i) \log p(x_i) \\
\sum_i^n p(x_i) = 1
$$

- 其中，定义$0log0=0$，使用极限定义$\lim_{x\to \infin} xlogx = 0$
- 使用拉格朗日乘子法获得 H(X)的最大值

$$
L(p, \lambda) = \sum_i^n p(x_i) \log p(x_i) + \lambda - \lambda\sum_i^n p(x_i) \\
\frac{\partial L}{\partial p(x_i)} =  \log p(x_i) + \frac{1}{\ln 2} - \lambda = 0
\rArr \lambda = \log p(x_i) + \frac{1}{\ln 2}
$$

对所有取值依次求偏导，发现 H(X)最大值(拉格朗日里的最小值)在均匀分布时取到。

- $H(X) \geq 0$

- ex:微分熵，定义在连续概率分布上的信息熵

$$
h(x) = -\int p(x) \log p(x) dx
$$

differential entropy 可以为负数，同时在均值和方差的连续分布当中，高斯分布具有最大的熵

#### 条件信息熵

定义为一个样本空间内，Y 事件发生时，X 事件发生的条件自信息期望

涉及到两个概率分布，因此需要对一个事件发生和所有事件发生的期望进行定义

一个事件发生时，X 分布的信息量期望

$$
H(X|y) = \mathbb{E}_{p(x|y)}[I(x | y)]=-\sum_x p(x|y) \log p(x|y)
$$

**Y 分布的事件发生时，X 分布的信息量的期望的期望**，引申[全期望公式](https://zhuanlan.zhihu.com/p/417592820)

$$
H(X|Y) = \sum_y p(y) H(X|y) = -\sum_y \sum_x p(xy) \log p(x|y)
$$

与条件互信息相同，表示的是 Y 分布对 X 分布贡献之后的信息量，其差值可以用另外一个函数表示，定义在 Topic 3。

#### 联合信息熵

定义为两个概率分布的联合自信息的期望

$$
H(X,Y) = \mathbb{E}[I(X,Y)] = - \sum_x \sum_y p(x,y) \log p(x,y)
$$

#### Prior Knowledge

1. 上凸函数/Concave Function
   $$
   \alpha f(x) + (1-\alpha)f(x) \leq f(\alpha x + (1-\alpha) x),\ \alpha \in [0,1]
   $$
2. Jensen 不等式
   若 f 严格上凸(等号仅取在$\alpha=0/1$或者$x_1=x_2$)，则

   $$
   \sum_k \lambda_k f(x_k) \leq f(\sum_k \lambda_k x_k), \ \sum_k \lambda_k = 1
   $$

   $proof$:

   1. $n=2$ 时，$\lambda_1 f(x_1) + \lambda_2 f(x_2) \leq f(\lambda_1 x_1 + \lambda_2 x_2)$, $\sum \lambda_i = 1$, 并且等号仅在 $\lambda_1 = 1, \lambda_2 = 0$或者$x_1 = x_2$时取到
   2. 假设对于 $n=k$ 时成立，那么对于 $n=k+1$ 时，要证明

      $$
      \sum_1^{k+1} \lambda_i f(x_i) \leq f(\sum_{i+1} \lambda_i x_i)
      $$

      即证明

      $$
      \begin{align}
         \sum_1^{k} \lambda_i f(x_i) + \lambda_{k+1} f(x_{k+1}) \leq f(\sum_1^{k} \lambda_i x_i + \lambda_{k+1} x_{k+1})
      \end{align}
      $$

      已知

      $$
      \sum_1^k \lambda_i = 1
      $$

      将 inequality 左边第一项转化为合一项，即

      $$
       \sum_1^{k} \lambda_i f(x_i) =
       \sum_1^k \lambda_i \sum_1^{k} \frac{\lambda_i}{\sum_1^k \lambda_i} f(x_i) \leq
      \sum_1^k \lambda_i f(\frac{\lambda_i}{\sum_1^k \lambda_i} x_i)
      $$

      = 当且仅当 $\lambda_i = 1$ 或者 所有$x_i$均相等时取等号

      于是(1)变为

      $$
      \begin{align}
         \sum_1^{k} \lambda_i f(x_i) + \lambda_{k+1} f(x_{k+1})
         \leq \sum_1^k \lambda_i f(\frac{\lambda_i}{\sum_1^k \lambda_i} x_i) + \lambda_{k+1} f(x_{k+1})
      \end{align}
      $$

      又因为$\sum_1^{k} \lambda_i + \lambda_{k+1} = 1$

      再使用一次 Jensen 不等式，得到

      $$
      \begin{align}
         \sum_1^k \lambda_i f(\frac{\lambda_i}{\sum_1^k \lambda_i} x_i) + \lambda_{k+1} f(x_{k+1})
         \leq f(\sum_1^k \lambda_i \frac{\lambda_i}{\sum_1^k \lambda_i} x_i + \lambda_{k+1} x_{k+1}) =
         f(\sum_1^{k+1} \lambda_i x_i)
      \end{align}
      $$

      = 当某一个$\lambda_i = 1$ 或者$\frac{\lambda_i}{\sum_1^k \lambda_i} x_i = x_{k+1}$相等时取等号

   **分析取等号条件**：
   当所有$x_i,i \leq k$相等，且$\frac{\lambda_i}{\sum_1^k \lambda_i} x_i = x_{k+1}$时取等号，可得所有的$x_i, i \leq k+1$相等时，取等号。

3. $\log x$是上凸函数,$E[\log x] \leq \log E[x]$

#### KL Divergence

若 P,Q 定义在同一个概率空间的不同测度，那么 KL Divergence 定义为

$$
D(P \| Q) = E_p [\log \frac{p(x)}{q(x)}] = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

**Properties**:

1.  KL Divergence 不是一个 metric/dist，因为 metric 需要满足以下性质（复习 mml，dist 可由 norm 确定:$d(x.y) = \| x-y \|$）

    1. 对称性
    2. 非负性
    3. 三角不等式

2.  可以用来描述概率分布的距离（但是必须定义在同一个概率空间之上）

3.  $D(P\|Q) \geq 0$, '=' iff $Q(x) = P(x)$

    $proof$:

$$
-D(P\|Q) = \sum_x p(x) \log \frac{q(x)}{p(x)} \leq^{\text{Jensen Inequality}}
\log \sum_x p(x)  \frac{q(x)}{p(x)} =
\log \sum_x q(x) = 0
$$

根据 Jensen 不等式的取等号条件，= iff $\frac{q(x)}{p(x)}$对所有$x$的均相等, 又因为概率归一，所以所有的$q(x) = p(x)$

#### Basic Properties

1. 熵不依赖分布的位置（大小）
2. 离散熵的非负性
3. 小概率事件对熵的影响很小
   $\lim_{\epsilon \to 0} - \epsilon \log \epsilon$ 因此，$\lim_{\epsilon \to 0} H(p_1,\dots, p_n - \epsilon, \epsilon) = H(p_1, \dots, p_n)$
4. $H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$
5. 离散熵的最大值取在均匀分布（证明见拉格朗日乘子法）
6. $H(X)$严格上凸
7. ==$H(Y|X) \leq H(Y)$==，等号当且仅当 $X \perp Y$
8. Chain Rule: ==$H(X_1, \dots, X_n) = \sum_i H(X_i | X_1, \dots, X_{i-1})$==
9. 联合熵不大于各自熵之和：==$H(X_1, \dots, X_n) \leq \sum H(X_i)$==,使用 7 和 8 可证明，等号当且仅当 $X_i \perp X_j, \forall i \neq j$

### Topic 3: Mutual Information

#### 平均互信息

集合$Y$ 与 事件 $x$ 的平均互信息定义为

$$
I(x;Y) = \mathbb{E}_{p(y|x)}[I(y)-I(y|x)] = \sum_y p(y|x) \log \frac{p(y|x)}{p(y)}
$$

平均互信息非负：$I(x;Y) = D(p(y|x) \| p(y)) \geq 0$

集合$Y$ 与 集合 $X$ 的平均互信息定义为

$$
I(X;Y) = \mathbb{E}_{p(x)}[I(x;Y)] = \sum_x p(x) \sum_y p(y|x) \log \frac{p(y|x)}{p(y)} = \\
\sum_x \sum_y p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

物理意义：$I(X;Y)$ 表示 $X$ 通过 $Y$ 获得的平均信息量

**性质**：

1. $I(X;Y) = I(Y;X) = H(X) - H(X|Y) = H(Y) - H(Y|X)$
2. $I(X;Y) \geq 0$, because $I(x;Y) \geq 0$
3. $I(X;Y) \leq H(X) / H(Y)$

#### 平均条件互信息

集合$Z$ 与 集合 $X$ 与 集合 $Y$ 的平均条件互信息定义为

$$
I(X;Y|Z) = \mathbb{E}_{p(z)}[I(X;Y|z)] = \\
\sum_z p(z) \sum_x \sum_y p(x,y|z) \log \frac{p(x,y|z)}{p(x|z)p(y|z)}
$$
