## 介绍

用于介绍 STL 中的常用容器和算法。应付只能使用 C++作答的考试。（**服了**）

```cpp
#include <bits/stdc++.h>
using namespace std;
```

## 容器

### Vector

用于代替数组，作为动态数组使用，相当于 python 中的 list。

#### 初始化

但是初始化较为麻烦, `vector<type> arr(length, [value])`。

```cpp
vector<int> arr;         // 构造int数组
vector<int> arr(100);    // 构造初始长100的int数组
vector<int> arr(100, 1); // 构造初始长100的int数组，初值为1

vector<int> mat[100];    // 构造100个vector<int>，用于邻接表的实现
vector<vector<int>> mat(100, vector<int> ());       // 构造初始100行，不指定列数的二维数组
vector<vector<int>> mat(100, vector<int> (666, -1)) // 构造初始100行，初始666列的二维数组，初值为-1
```

#### 操作

- `arr.size()`：返回数组长度
- `arr.push_back(value)`：在数组末尾添加元素
- `arr.pop_back()`：删除数组末尾元素
- `arr.empty()`：判断数组是否为空
- `arr.resize(length, [value])`：调整数组长度
- `arr.clear()`：清空数组

#### 替换二维数组

如果我们要使用一个常数定义的二维数组`int mat[100010][100010]`，可以使用`vector<vector<int>> mat(n+10, vector<int> (m+10))`来代替。

### queue

- `queue<type> q`：定义一个队列
- `q.push(value)`：入队
- `q.pop()`：出队(注意没有返回值)
- `q.front()`：返回队首元素
- `q.back()`：返回队尾元素
- `q.empty()`：判断队列是否为空

### priority_queue

`priority_queue<type, docker, comp> pq`

- `type`：数据类型
- `docker`：容器类型，默认为`vector<type>`
- `comp`：比较函数，默认为`less<type>`

```cpp
priority_queue<int> pq; // 默认为大根堆
priority_queue<int, vector<int>, greater<int>> pq; // 小根堆
```

- `pq.push(value)`：入队
- `pq.pop()`：出队
- `pq.top()`：返回队首元素
- `pq.empty()`：判断队列是否为空
- `pq.emplace(value)`：入队，区别在于`push`是拷贝入队，`emplace`是直接入队

### unordered_map

- `unordered_map<type1, type2> mp`：定义一个哈希表
- `mp[key] = value`：插入键值对
- `mp.find(key)`：查找键值对
- `mp.erase(key)`：删除键值对
- `mp.size()`：返回哈希表长度
- `mp.empty()`：判断哈希表是否为空
- `mp.clear()`：清空哈希表
- `mp.count(key)`：返回键值对的个数
- `mp[key]`：返回键值对的值

### unordered_set

- `unordered_set<type> st`：定义一个哈希集合
- `st.insert(value)`：插入元素
- `st.erase(value)`：删除元素
- `st.count(value)`：返回元素个数
- 

## 算法

### 排序, sort

- `sort(arr.begin(), arr.end())`：默认升序排序
- `sort(arr.begin(), arr.end(), comp)`：自定义比较函数
- `sort(arr.begin(), arr.end(), greater<type>())`：降序排序
