## Attention is All You Need(自学用)

### Background

Google Brain 团队提出的模型，全程只用到了 LN, FC, Attention，没有用到 CNN 和 RNN，是一个纯 Transformer 模型。

而在之前，由于使用的 序列化模型，比如 RNN，LSTM，GRU，无法并行，在长序列数据中训练时间难以接受。

而 Transformer 模型，可以并行计算，同时采用 encoder-decoder 结构，可以处理序列到序列的问题，在机器翻译上表现优异。

在此之前就已经有了 Attention 机制，但是在 RNN 中 Attention 机制是一个附加的模块，而在 Transformer 中 Attention 是核心模块。

同时为了模拟 CNN 能够捕捉多样化的特征，Transformer 使用了 Multi-head Attention。

### Model Architecture

![img](https://img2023.cnblogs.com/blog/3436855/202405/3436855-20240521012140163-1540593826.png)

inputs 是一个序列$x = (x_1, x_2, \cdots, x_n)$, outputs 是一个序列$y = (y_1, y_2, \cdots, y_m)$, embedding 层每一个 word 输出一个连续的嵌入向量$z = (z_1, z_2, \cdots, z_l)$。

_Mark: 输入输出序列长度不一致是合理的，因为 seq2seq 本身就是不一致的_

使用 Auto-regressive 的方式，即在预测第$i$个词时，只使用前$i-1$个词，而不使用后面的词。

#### Encoder

Encoder 由 $N = 6$ 个相同的层组成，每一层包含两个子层：
