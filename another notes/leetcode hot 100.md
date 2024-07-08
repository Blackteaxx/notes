### unordered_set

#### 最长连续序列

使用哈希表做，首先将所有元素放入哈希表中，然后遍历哈希表，对于每一个元素，如果它的前一个元素不在哈希表中，那么就开始计算连续序列的长度。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> num_set;

        for (int &num: nums) {
            num_set.insert(num);
        }

        int length = 0;

        for (int &num: nums) {
            if (!num_set.count(num-1)) {
                int j = 1;
                for (; num_set.count(num+j)!=0; j++);
                length = max(length, j);
            }
        }

        return length;
    }
};
```

### 二分查找

区间查找模板

```cpp

// mid 在右半区
int SR(vector<int>& nums, int target) {
    int low = 0; int high = nums.size()-1;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (nums[mid] < target) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
}
```

```cpp
// mid 在左半区
int SL(vector<int>& nums, int target) {
    int low = 0; int high = nums.size()-1;
    while (low < high) {
        int mid = (low + high + 1) / 2;
        if (nums[mid] <= target) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
}
```

### 矩阵操作

#### 48. 旋转图像（该死的 pdd 一面）

就是将矩阵逆时针旋转，不难看出，这个操作可以分解为两个操作：**转置**和**左右翻转**

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        // 沿着副对角线交换元素

        int n = matrix.size();

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n-1-i; j++) {
                swap(matrix[i][j], matrix[n-j-1][n-i-1]);
            }
        }

        for (int j = 0; j < n; j++) {
            int start = 0;
            int end = n-1;
            while (start < end) {
                swap(matrix[start][j], matrix[end][j]);
                start++; end--;
            }
        }
    }
};
```

#### 240. 搜索二维矩阵 II

朴素做法：

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int col = matrix[0].size();
        for (int i=0; i<matrix.size(); i++) {
            for (int j =0; j<col; j++) {
                if (matrix[i][j] == target) {
                    return true;
                } else if (matrix[i][j] > target) {
                    col = j;
                    break;
                }
            }
        }
        return false;
    }
};
```

从右上角开始看，如果类似于一个 BST

![img](https://img2023.cnblogs.com/blog/3436855/202406/3436855-20240628213659311-175284958.png)

即如果当前元素大于 target，则向`i++`, 如果当前元素小于 target，则向`j--`。

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int j = matrix[0].size() - 1;
        int i = 0;

        while (i >= 0 && j >= 0 && i < matrix.size() && j < matrix[0].size()) {
            if (matrix[i][j] < target) {i++;}
            else if (matrix[i][j] > target) {j--;}
            else {return true;}
        }
        return false;
    }
};
```

### 回溯/全排列

#### 39. 组合总和

需要注意的是，需要**判断组合的重复性**，在遍历过程中，我发现这道题与**求幂集**有点类似，因为重复性的解决方案是按照元素来遍历，而不是按照组合来遍历，这样就可以避免重复的组合。

```cpp
class Solution {
public:
    vector<vector<int>> res;
    vector<int> t;

    void dfs(vector<int>& candidates, int target, int cur) {
        if (target == 0) {
            res.push_back(t);
            return;
        }
        if (cur == candidates.size() || target < 0) {
            return;
        }

        // 按照元素来遍历，若当前元素的倍数小于等于target，则继续遍历
        for (int i=0; candidates[cur] * i <= target; i++) {
            for (int j=0;j<i;j++) {t.push_back(candidates[cur]);}
            dfs(candidates, target-candidates[cur]*i, cur+1);
            for (int j=0;j<i;j++) {t.pop_back();}
        }

    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        dfs(candidates, target, 0);
        return res;
    }
};
```
