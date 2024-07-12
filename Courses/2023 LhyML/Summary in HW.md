## Tensor Board

[tutorial 1](https://zhuanlan.zhihu.com/p/471198169)
[tutorial 2](https://blog.csdn.net/qq_41656402/article/details/131123121)

## HW 1: Regression

回归任务，预测 COVID-19 的病例数目，使用前三天的数据预测第四天的数据

使用`MSE`作为损失函数，需要进行的工作有

- run baseline model
- modify the architecture of the neural network
- implement the improved optimization algorithm and L2 regularization
- implement the feature selection algorithm

### Baseline Model

![img](https://img2023.cnblogs.com/blog/3436855/202407/3436855-20240712200424250-316693650.png)

![img](https://img2023.cnblogs.com/blog/3436855/202407/3436855-20240712200538861-1346069606.png)

实现了一个简单的全连接神经网络，架构为`input_dims -> 16 -> 8 -> 1`，使用`MSE`作为损失函数，使用`SGD`作为优化器，并且使用`early stopping`来防止过拟合

- early stopping: 即在验证集上的 loss 不再下降一定次数后，停止训练，防止过拟合

设置的参数有:

- epochs = 5000
- batch_size = 256
- learning_rate = 1e-5
- early_stop = 600

运行时间为`5min 30s`, mean squared error 为`2.6849`

### Modify the Architecture

由于是一个线性回归，也没有太多的特征，所以可以尝试使用更简单的模型，或更加深层的模型，我试了两种模型

- 更简单的模型: `input_dims -> 8 -> 1`，运行时间为`8min`, mse 为`1.8476`
- 更深层的模型: `input_dims -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1`, 运行时间为`8min`, mse 为`41.2422`

值得注意的是，使用一层直接 loss 变成 `nan`了

### Improved Optimization Algorithm

主要使用了`AdamW`来代替`SGD`, 并且添加了`L2 regularization`，设置`weight_decay = 0.01`

运行时间为`9min`, mse 为`1.3608`

至于 feature selection algorithm, 没有实现，因为是传统的特征选择算法，不清楚如何 expand to neural network

## HW 2: Classification
