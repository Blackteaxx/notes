## 用户活跃指标

**北极星指标**

- 用户规模
  - DAU、MAU
- 消费
  - 人均使用推荐的时长、人均阅读笔记的数量（**更能体现推荐系统的性能**）
- 发布
  - 发布渗透率、人均发布量

## 实验流程

离线实验 --> 小流量 A/B 测试 --> 全流量上线

- 离线实验
  使用历史数据上做训练、测试，但是没有真实用户参与，无法真实反映用户的行为，北极星指标也无法体现
- 小流量 A/B 测试
  部署到实际产品，使用用户实际的反应来评估推荐效果

## 推荐系统的链路

![img](https://img2023.cnblogs.com/blog/3436855/202406/3436855-20240604143410033-1683453528.png)

### 简单概念介绍

可以看作一个漏斗，逐步将大量的文档进行筛选，最终推荐给用户

- 召回

  - 从海量的物品中选取一部分候选集
  - 召回的目的是尽可能多的覆盖用户的兴趣

- 粗排
  - 用规模小的模型对召回的结果进行粗排，选定一部分候选集
- 精排
  - 用规模大的模型对粗排的结果进行精排，选定最终的推荐结果
- 重排
  - 对最终的推荐结果进行排序，选取最终的推荐结果
  - 计算多样性分数，最后插入广告...

### 具体介绍

召回：会有多个召回策略，协同过滤、双塔模型，每一个召回策略都会有一个召回结果

- **粗排**：给所有的笔记打分，选取分数最高的笔记。粗排的输入包括用户的特征、物品特征、统计特征等

精排：给粗排的结果打分，选取分数最高的笔记

（用于平衡计算量和准确性）， 使用 NN 对指标进行预测，然后对结果进行排序

重排：对精排的结果挑选最终展示给用户的笔记，做多样性抽样（MMR、DPP）。并使用规则打散相似的笔记，有助于根据生态要求排列笔记

## 推荐系统中的 A/B Test

- 实现了一种召回通道，离线实验结果正常
- 下一步做线上的小流量 A/B 测试，考察新的召回通道对指标的影响
- 模型中还有一些参数，需要通过 A/B 测试来确定最优的参数，例如有$\{ 1,2,3 \}$

**随机分桶**

将所有的用户比如说按照平均的指标相同划分为 10 个桶，然后随机选择 4 个桶，其中 3 个桶使用新的召回通道，1 个桶使用原来的召回通道，然后对比指标的变化是否存在显著性

**流量不够用怎么办？**

使用分层实验，同层的实验**使用不同/互斥的桶**，不同层的实验**使用独立随机/正交的桶**

因为同层的实验天然互斥、一种用户只能进入一种实验，同时会相互干扰

而不同层的实验则不会相互干扰(1+1=2)，可以同时进行

**Holdout 机制**

将所有的用户保留一个桶作为 holdout 组，不参与实验，用来评估实验的效果

能够保证每一个考核周期都有一个对照组，用来评估实验的效果