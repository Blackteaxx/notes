## Attention

### Vector Set as Input

如果我们要处理一个序列输入，比如一个句子，是由一些 words 组成的，使用 embedding 的方式，我们可以得到一个向量序列

声音信号可以被转化为一个向量序列；图数据中的节点可以被转化为一个向量序列

### Output

1. 每一个向量都有一个标签输出，如：词性标注

2. 整个序列有一个标签输出，如：情感分类，语者识别

3. 一个向量序列，输出一个向量（seq2seq），如：机器翻译

这节课讨论的是第一种情况，Sequence Labeling

### Sequence Labeling

输入：一个向量序列 $x_1, x_2, ..., x_n$

输出：一个标签序列 $y_1, y_2, ..., y_n$

- 首先的想法是使用一个 FCN，将每一个向量映射到一个标签，但是这样的方法忽略了序列中的信息，同时比如在词性分类中，一个词的词性可能和前后的词有关，FCN 无法捕捉到这种信息

因此，我们需要一种方法，能够考虑到序列中的信息，可以采用有 window 的方法，但是这样的方法需要人为设定 window 的大小，不够灵活

### Self-Attention

**Self-attention**: 考虑所有的上下文信息，不需要人为设定 window 的大小，可以自动学习到序列中的信息，最终输出是加权求和的结果

包含有三个向量：Query, Key, Value

$$
Q = W_q \cdot X \\
K = W_k \cdot X \\
V = W_v \cdot X
$$

1. Attention Score: $e_{ij} = a(q_i, k_j)$ 用来衡量 $q_i$ 和 $k_j$ 之间的相关性，有 dot product, Additive 等方式

   1. dot product: $e_{ij} = (W_q \cdot q_i)^T \cdot W_k \cdot k_j$
   2. Additive: $e_{ij} = v^T tanh(W_1 q_i + W_2 k_j)$

   因此我们可以得到一个 $n \times n$ 的矩阵 $E$，其中 $E_{ij} = e_{ij}$, $E = (W_q X)^T W_k X$

2. Attention Weight: $a_{ij} = \frac{exp(e_{ij})}{\sum_{j=1}^n exp(e_{ij})}$, 即每一个 query 进行 softmax 归一化

3. Attention Output: $o_i = \sum_{j=1}^n a_{ij} V_j$，即每一个 query 的输出是 value 的加权求和，$O = A V^T$

可以看到，Attention 的计算是并行的，可以使用矩阵乘法来加速计算

重新陈述 Attention 过程：

$$
X_{m\times n} = [x_1, x_2, ..., x_n] \\
$$

1. 计算 Query, Key, Value

   $$
   Q = W_q \cdot X \\
   K = W_k \cdot X \\
   V = W_v \cdot X
   $$

2. 计算 Attention Score

   $$
   E = (W_q X)^T W_k X
   $$

3. 计算 Attention Weight(row-wise)

   $$
   A = softmax(E)
   $$

4. 计算 Attention Output

   $$
   O = A V^T
   $$

需要学习的参数有 $W_q, W_k, W_v$

### Multi-Head Attention

想要让 attention 像 CNN 一样，一层可以输出多个 channel，即多种多样的特征，因此引入了 Multi-Head Attention

具体的做法是将 query，key，value 额外进行一次线性变换，总共有 $h$ 个 head，每一个 head 有自己的 $W_q, W_k, W_v$

$$
Q_i = W_{qi} \cdot X \\
K_i = W_{ki} \cdot X \\
V_i = W_{vi} \cdot X
$$

然后将每一个 head 的输出进行拼接，再进行一次线性变换

### Positional Encoding

在 attention 中，考虑相关性，但是没有考虑到位置信息，因此引入了 Positional Encoding

### Truncated Self-Attention

在实际应用中，序列可能会很长，计算复杂度会很高，因此引入了 Truncated Self-Attention，即只考虑一部分的序列，即只计算 $k$ 个位置的相关性，加快计算速度

### Self-Attention for image

在图像中，一个像素（包含所有 channel）可以看作是一个向量，因此可以使用 Self-Attention 来处理图像

- Self-Attention and CNN：Self-Attention 可以捕捉到全局信息，CNN 可以捕捉到局部信息，同时 attention 是可学习的

An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

### RNN vs. Self-Attention

RNN：逐个处理序列，不能并行计算，不能捕捉到长距离的依赖关系

Self-Attention：可以并行计算，可以捕捉到长距离的依赖关系

### Attention For Graph

在图数据中，节点之间的关系是很重要的，因此可以使用 Self-Attention 来处理图数据，其中的相关性可以通过节点之间的关系来计算

![img](https://img2023.cnblogs.com/blog/3436855/202408/3436855-20240819214521725-1891110775.png)
