## Ch2 知识表示

### What is knowledge representation?

Knowledge: DIKW (Data, Information, Knowledge, Wisdom)

Knowledge Representation: the process of **encoding knowledge into a form/architecture** that can be used by **a computer**.

Way of representing knowledge:

- Logic-based representation
- Production representation
- Architectural representation: Semantic Networks, Frames
- 过程表示法

### 一阶谓词逻辑表示法

强调 表达式 **truth-preserving** operation

#### 基础概念

回忆离散数学里数理逻辑内容：

- 命题
- 真值
- 论域: 由所讨论对象的全体构成的集合
- 个体: 论域中的元素,表示独立存在的事物或概念(可以是变元/常量)
- 谓词：**表示个体间关系的符号**，是一个函数

  设$D$为论域，$P$为谓词，$x$为个体，$P: D^n \to \{ T, F \} $是一个映射，其中

  $$
  D^n = \{(x_1, x_2, \dots, x_n) | x_i \in D \}
  $$

- 函数：**表示个体间的映射关系**，是一个函数

  设$D$为论域，$f$为函数，$x$为个体，$f: D^n \to D$是一个映射，其中

  $$
  D^n = \{(x_1, x_2, \dots, x_n) | x_i \in D \}
  $$

- 连词
- 量词
- 项
- 原子谓词公式
- 合式公式
- 辖域、辖域变元、自由变元
- 变元替换：需要注意换名不能替换成已有的变元名

#### 知识表示的步骤

例： 表示知识“所有教师都有自己的学生”

1. 确定论域：人
2. 确定谓词：教师、学生、教
3. 表示知识：$\forall x \exists y \text{ 教师}(x) \to  \text{ 学生}(y) \land \text{教}(x, y)$

#### 应用

机器人移动盒子的例子

需要定义**谓词**、**Domain**、以及一系列改变状态的**操作**

使用谓词解修道士与野人问题

### 产生式表示法

**逻辑表示只强调真实性**，忽略了前提与结论间的特定关系

是使用最多的知识表示方法

- 事实： 断言一个语言变量的值或断言多个语言变量之间关系的陈述句
  例子： `John is a student`， `John is a good student`
- 事实的表示
  三元组： （对象，属性， 值 ）或（关系，对象 1，对象 2）
- 规则： 描述事物之间的因果关系，由条件和结论组成的陈述句
  例子： `IF John is a student THEN John is a good student`

由于规则的不确定性，产生式系统的推理过程是**不确定的**，同时可能会产生**冲突**

其求解过程是一种反复进行的“**匹配—冲突消解—执行**”过程

### 语义网络

语义网络是一种有向图, 用**实体及其语义关系来表达知识**。
**结点代表实体**，表示各种事物、概念、属性、状态、事件、动作等；
**弧代表语义关系**，表示连结的实体间的语义联系，**它必须带有标识**。

基本语义关系有：

- ISA：表示实体与其类别的关系
- AKO：表示实体与其上位类的关系
- HAVE：表示实体与其属性的关系

![img](https://img2023.cnblogs.com/blog/3436855/202406/3436855-20240601185826922-945873881.png)

表达全称量词$\forall$, 需要网络分区，即使用一个概念节点来表示全称量词的范围

![img](https://img2023.cnblogs.com/blog/3436855/202406/3436855-20240601190031306-1918799117.png)

### Knowledge Graph

The Knowledge Graph is a **knowledge base** used by Google to enhance its search engine's search results with semantic-search information gathered from a wide variety of sources. 