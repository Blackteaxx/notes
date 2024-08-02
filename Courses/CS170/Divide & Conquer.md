## Fibonacci Sequence

$$
F_n = \begin{cases}
0, & n = 0 \\
1, & n = 1 \\
F_{n-1} + F_{n-2}, & n > 1
\end{cases}
$$

4 ways to calculate the Fibonacci sequence:

### 1. Recursion

```python
def Fib(n):
    if n <= 1:
        return n
    else:
        return Fib(n-1) + Fib(n-2)
```

By **counting the flops**(floating point ops), we can find that the time complexity. Use $T(n)$ to denote the time complexity of calculating the $n$-th Fibonacci number, then we have

$$
T(n) = \begin{cases}
    0, & n <= 1 \\
    T(n-1) + T(n-2) + 1, & n > 1
\end{cases}
$$

It is similar to the fibonacci sequence itself, and we can find that $F_n \geq 2^{\frac{n}{2}}$, so the time complexity $T(n) \geq 2^{\frac{n}{2}} + 1$

### 2. Iteration

```python
def FasterFib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for i in range(2, n+1):
        a, b = b, a+b
    return b
```

The time complexity of this algorithm is $\mathcal{O}(n)$

### 3. Fast Matrix Powing

Define a matrix:

$$
\begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}
$$

Then we have

$$
\begin{bmatrix}
F_{n+1} \\
F_n
\end{bmatrix} = \begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix} \begin{bmatrix}
F_n \\
F_{n-1}
\end{bmatrix}
$$

so we can get

$$
\begin{bmatrix}
F_{n+1} \\
F_n
\end{bmatrix} = \begin{bmatrix}
1 & 1 \\
1 & 0
\end{bmatrix}^n \begin{bmatrix}
1 \\
0
\end{bmatrix}
$$

We can get F_n by calculating the power of matrix directly, and the time complexity is $\mathcal{O}(n)$

but the power of number, we can **use left shift to calculate the power of number**, so the time complexity is $\mathcal{O}(\log n)$, for example, $2^{10} = 2^{1010} = 2^8 \cdot 2^2$, it is called repeated squaring.

So we can calculate the power of matrix by repeated squaring, and the time complexity is $\mathcal{O}(\log n)$**(What?)**

### 4. Constant Time(?)

We can use the formula to calculate the Fibonacci number directly by **doing the eigenvalue decomposition of the matrix**

$$
A^n = P \begin{bmatrix}
\lambda_1^n & 0 \\
0 & \lambda_2^n
\end{bmatrix} P^{T}
$$

where $\lambda_1 =  \frac{1 + \sqrt{5}}{2}$, $\lambda_2 =  \frac{1 - \sqrt{5}}{2}$

so we get the formula

$$
F_n = \frac{\lambda_1^n - \lambda_2^n}{\lambda_1 - \lambda_2}
$$

### Summary

$F_n = \frac{\lambda_1^n - \lambda_2^n}{\lambda_1 - \lambda_2} \approx \exp(n)$, so the digits of the Fibonacci number is $\mathcal{O}(\log F_n) = \mathcal{O}(n)$

And the **time complexity of addition** is $\mathcal{O}(d)$, multiplication is $\mathcal{O}(d^2)$

So the time complexity of calculating the Fibonacci are like following:

| Method    | Time Complexity          |
| --------- | ------------------------ |
| Recursion | $\mathcal{O}(\exp(cn))$  |
| Iteration | $\mathcal{O}(n^2)$       |
| Matrix    | $\mathcal{O}(n^2\log n)$ |

## $\mathcal{O}, \Omega, \Theta$

- $f, g: \mathbb{Z}^+ \to \mathbb{Z}^+$
- $f = \mathbb{O}(g) \iff \exist c>0, \forall n, f(n) \leq c \cdot g(n)$(can be seen as the upper bound)
- $f = o(g) \iff \lim_{n \to + \infin} \frac{f(n)}{g(n)} = 0$
- $f = \Omega(g) \iff g = \mathcal{O}(f)$

## Master Theorem

The master theorem is a formula for **solving recurrence relations** of the form

$$
T(n) = aT(\frac{n}{b}) + c \cdot n^d
$$

we expand the formula to

$$
T(n) = c \cdot n^d \cdot (1 + \frac{a}{b^d} + (\frac{a}{b^d})^2 + \cdots + (\frac{a}{b^d})^{\log_b n}) \\
= \begin{cases}
\mathcal{O}(n^d), & a < b^d \\
\mathcal{O}(n^d \log n), & a = b^d \\
\mathcal{O}(n^{\log_b a}), & a > b^d
\end{cases}
$$

So we can find the asymptotic complexity actually **is determined by the the first/last/all the intern terms** when calculating **the geometric series**.

## Matrix Multiplication

$$
A_{n \times n} \cdot B_{n \times n} = C_{n \times n}
$$

### 1. Naive Method

```python
def NaiveMatrixMultiplication(A, B):
    n = len(A)
    C = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C
```

the asymptotic complexity is $\mathcal{O}(n^3)$

### 2. Divide and Conquer

We can split the matrix into 4 subparts, and calculate the product of the 4 parts, and then add them together.

$$
\begin{bmatrix}
A_{11} & A_{12} \\
A_{21} & A_{22}
\end{bmatrix} \cdot \begin{bmatrix}
B_{11} & B_{12} \\
B_{21} & B_{22}
\end{bmatrix} = \begin{bmatrix}
C_{11} & C_{12} \\
C_{21} & C_{22}
\end{bmatrix}
$$

so we have

$$
C_{11} = A_{11} \cdot B_{11} + A_{12} \cdot B_{21} \\
C_{12} = A_{11} \cdot B_{12} + A_{12} \cdot B_{22} \\
C_{21} = A_{21} \cdot B_{11} + A_{22} \cdot B_{21} \\
C_{22} = A_{21} \cdot B_{12} + A_{22} \cdot B_{22}
$$

Runtime Analysis:

$$
T(n) = 8 \cdot T(\frac{n}{2}) + c \cdot n^2
$$

According to the master theorem, $a = 8, b=2, d=2$, and $8 > 2^2$, so the asymptotic complexity is $\mathcal{O}(n^{\log_2 8}) = \mathcal{O}(n^3)$.

### 3. Strassen Algorithm

We can calculate the product of two $n \times n$ matrix by 7 multiplications

$$
P_1 = A_{11} \cdot (B_{12} - B_{22}) \\
P_2 = (A_{11} + A_{12}) \cdot B_{22} \\
P_3 = (A_{21} + A_{22}) \cdot B_{11} \\
P_4 = A_{22} \cdot (B_{21} - B_{11}) \\
P_5 = (A_{11} + A_{22}) \cdot (B_{11} + B_{22}) \\
P_6 = (A_{12} - A_{22}) \cdot (B_{21} + B_{22}) \\
P_7 = (A_{11} - A_{21}) \cdot (B_{11} + B_{12})
$$

and then we can get the product of two $n \times n$ matrix by

$$
C = \begin{bmatrix}
P_5 + P_4 - P_2 + P_6 & P_1 + P_2 \\
P_3 + P_4 & P_1 + P_5 - P_3 - P_7
\end{bmatrix}
$$

So the runtime analysis is

$$
T(n) = 7 \cdot T(\frac{n}{2}) + c \cdot n^2 = \mathcal{O}(n^{\log_2 7}) = \mathcal{O}(n^{2.81})
$$

## Sort

### Merge Sort

$$
A = [A_1, A_2, \cdots, A_{\frac{n}{2}},\cdots, A_n]
$$

$$
T(n) = 2T(\frac{n}{2}) + \mathcal{O}(n) = \mathcal{O}(n \log n)
$$

### Iteratively Merge Sort

**Construct a Queue** to store the subarray, and then **merge first two subarrays**, add the merged subarray to the Queue, and then merge the next two subarrays, and so on.

### Random Algothrims

- Las Vegas: the output is always correct, but the runtime is random
- Monte Carlo: the output is random, but the runtime is always correct

### The lower bound of sorting based on comparison model - Decision Tree

- Input: Unknown Permutation of $1, 2, \cdots, n$
- Output: Determined Permutation of $1, 2, \cdots, n$

The Decision Tree has $n!$ leaves **at least**(because the algothrim may do some redunt comparisions), the height of the tree is the worst case runtime of the sorting algorithm.

And we know that **the least height of the binary tree given** $n!$ leaves is $\log_2 n! = \Omega(n \log n)$

## Median/Selection

Output the smallest $k=\frac{n}{2}$ element in an array

### 1. Quick Select

- Choose a pivot
- Partition the array into two parts
- Recursively select the $k$-th element in the left/right part

but runtime analysis may be by the help of Probabilistic Method

$$
T(n) = \begin{cases}
n, & n = 1 \\
T(\frac{n}{i}) + cn, & n > 1
\end{cases}
$$

### 2. Deterministic Quick Selection

Split the array into $\frac{n}{5}$ groups, and then find the median of each group by sorting($5\log 5$), and then find the median of the medians($\frac{n}{5}\log \frac{n}{5}$), and then partition the array into two parts, and then recursively select the $k$-th element.

- Break the array into $\frac{n}{5}$ groups
- $B \gets \text{array of medians in each group}$
- $p \gets \text{median of } B$
- Partition the array into two parts, $L \gets \{< p\}$, $R \gets \{> p\}$
- if $k = | L | + 1, \text{return } p$
- if $k < | L | + 1, \text{return } \text{Select}(L, k)$
- if $k > | L | + 1, \text{return } \text{Select}(R, k - | L | - 1)$

and the length of $|L|, |R|$ depend on the property of the median of medians

$$
|L| \geq \frac{3}{10}n, | R | \leq \frac{7}{10}n \\
|R| \geq \frac{3}{10}n, | L| \leq \frac{7}{10}n
$$

runtime analysis:

$$
T(n) \leq \mathcal{O}(n) + \frac{n}{5} \mathcal{O}(1) + T(\frac{n}{5}) + \mathcal{O}(n) + T(\frac{7}{10}n) \leq \\
\mathcal{O}(n) + T(\frac{7}{10}n) + T(\frac{n}{5})
$$

Guess $T(n) \leq B \cdot n$, then we try to prove it via induction

- Base Case: $T(1) = 1 \leq B$
- Inductive Step: $T(n) \leq B \cdot n$, then we have $B \cdot \frac{7}{10}n + B \cdot \frac{n}{5} + cn \leq B \cdot n \Rightarrow \frac{B}{10} \geq c$
