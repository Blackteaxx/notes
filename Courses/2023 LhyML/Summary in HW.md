## Tensor Board

[tutorial 1](https://zhuanlan.zhihu.com/p/471198169)
[tutorial 2](https://blog.csdn.net/qq_41656402/article/details/131123121)

在 terminal 中输入以下命令，启动 tensorboard

```shell
tensorboard --logdir=runs
```

在 python 中导入 tensorboard 的库，使用方法如下

```python
from torch.utils.tensorboard import SummaryWriter

# 创建一个writer对象，log_dir是保存日志的路径
writer = SummaryWriter(log_dir='runs/')

train_loss = sum(train_loss) / len(train_loss)
train_acc = sum(train_accs) / len(train_accs)

# Print the information.
print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

# write to tensorboard
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
```

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

分类任务，将一个音频帧分类为 41 个不同 phoneme

### 数据描述

- train_labels.txt: 每一行是一个音频的标签，包含有文件名和每一帧对应的 phoneme

### 需要进行的工作有

- run baseline model, test the ability of sample model
- modify the architecture of DNN
- implement batch normalization and dropout

### Baseline Model

一个多层的 Softmax 回归网络，架构是堆叠 Basic Block 的形式，每个 Basic Block 包含一个全连接层和一个 ReLU 激活函数，`input_dims -> output_dims`，堆叠了`hidden_layer`层，隐藏层的维度为`hidden_dims`

- 合并音频帧的特征，`concat_nframes = 3`
- 使用`Adam`作为优化器
- 使用`CrossEntropy`作为损失函数
- `num_epoch = 10`
- `learning_rate = 1e-4`
- `hidden_layers = 2`
- `hidden_dim = 64`

最终 Validation Accuracy 为`0.5041`

### Medium Model

主要修改了网络的架构，即 concat 的长度，以及隐藏层维度与层数

1. Medium Model 1: `concat_nframes = 11`, `hidden_layers = 5`, `hidden_dim = 64`, `num_epoch = 10`, Validation Accuracy 为`0.58435`

2. Medium Model 2: `concat_nframes = 15`, `hidden_layers = 15`, `hidden_dim = 128`, `num_epoch = 20`, Validation Accuracy 为`0.62155`

### Strong Model

1. Strong Model 1

在 Medium Model 2 的基础上，添加了`Batch Normalization`、`Dropout`、`L2 Regularization`和余弦退火学习率调度

- `batch_size = 512`
- `dropout = 0.5`
- `weight_decay = 0.01`
- `scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=8,T_mult=2,eta_min = learning_rate/2)`

Validation Accuracy 为`0.42888`

2. Strong Model 2

Validation Accuracy 为`0.84554`

## HW 3: Image Classification

图像分类任务，共有 11 类

### 数据描述

- ./train (Training set): 图像命名的格式为 “x_y.png”，其中 x 是类别，含有 10,000 张被标记的图像
- ./valid (Valid set): 图像命名的格式为 “x_y.png”，其中 x 是类别，含有 3,643 张被标记的图像
- ./test (Testing set): 图像命名的格式为 “n.png”，n 是 id，含有 3,000 张未标记的图像

### 操作简述

1. Data Augmentation

`import torchvision.transforms as transforms`，使用`transforms.Compose`来组合多种数据增强方式，如`transforms.RandomHorizontalFlip`、`transforms.RandomRotation`、`transforms.RandomResizedCrop`等

通过如下的操作使用到了图像的数据增强

```python
# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You may add some transforms here.

    # 几何变换
    transforms.RandomHorizontalFlip(0.5), # 随机横向翻转
    transforms.RandomVerticalFlip(0.5), # 随机竖向翻转
    transforms.RandomApply(transforms=[
        transforms.RandomRotation(degrees=(0, 60))],
                           p=0.6), # 随机旋转
    transforms.RandomAffine(30), # 随机仿射

    # 像素变换
    transforms.RandomGrayscale(p=0.2), # 随机灰度化，p为灰度化的概率


    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
])

def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("\\")[-1].split("_")[0])
        except:
            label = -1 # test has no label

        return im,label
```

这样操作保证了在每一个 epoch 中，每一个 batch 的数据都是不同的，增加了模型的泛化能力

2. Model

采用`ResNet`跳连接方式，首先尝试自己实现一个`ResNet`，然后使用`torchvision.models.resnet18`来实现
