## Algothrim

start from basic arithmetic

### 1. Addition

$\mathcal{O}(n)$, pass from right to left, and carry the 1 to the next digit. And at least we must read them into the memory, so the best cost is $\mathcal{O}(n)$

### 2. Multiplication

1. $\mathcal{O}(n^2)$ school method

2. divide and conquer

   $$
    x = x_h \cdot 10^{\frac{n}{2}} + x_l \\
    y = y_h \cdot 10^{\frac{n}{2}} + y_l \\
    x \cdot y = x_h \cdot y_h \cdot 10^n + (x_h \cdot y_l + x_l \cdot y_h) \cdot 10^{\frac{n}{2}} + x_l \cdot y_l
   $$

   so, the cost of run time, we denote $T(n)$ as the cost of multiplication of two $n$-digit numbers, then we have

   $$
    T(n) = \begin{cases}
    1, & n = 1 \\
    4T(\frac{n}{2}) + c \cdot n, & n > 1
    \end{cases}
   $$

   the **recurrence tree** is the tool to analyze the run time of divide and conquer algorithm

   $$
   T(n) = 2^{\log_2 n + 1} \cdot n = \mathcal{O}(n^2)
   $$

3. improved divide and conquer(Karatsuba algorithm)

   $$
    x = x_h \cdot 10^{\frac{n}{2}} + x_l \\
    y = y_h \cdot 10^{\frac{n}{2}} + y_l \\
    x \cdot y = x_h \cdot y_h \cdot 10^n + ((x_h + x_l) \cdot (y_h + y_l) - x_h \cdot y_h - x_l \cdot y_l) \cdot 10^{\frac{n}{2}} + x_l \cdot y_l
   $$

   namely, we can calculate the cross product of $x_h + x_l$ and $y_h + y_l$ to **get the middle term**

   $$
    T(n) = 3T(\frac{n}{2}) + c \cdot n
   $$

   $$
   T(n) = \frac{(\frac32)^{\log_2 n +1}-1}{\frac32 - 1} = \mathcal{O}(3^{\log_2 n}) = \mathcal{O}(n^{\log_2 3})
   $$

## Big-O Notation

Big O notation is a way that can describe the **asymptotic complexity** of a function when the argument tends towards a particular value or infinity. It is used to describe the **upper bound** of the function.

> **Definition**
>
> $$
> f(n) = \mathcal{O}(g(n)) \\
> \text{ if } \exists c > 0, N \in \mathbb{N}, \forall n > N, 0 < f(n) \leq c \cdot g(n)
> $$

**An insightful analysis is based on the right simplifications**. When we analyze the concrete algorithm, we should do some **simplification**, for example, we can **assume that all the basic computer operations are the same cost**, and we can ignore the lower order terms and the constant factors.

We can use the **inequality of limits**(in mathematical ananlysis) to prove the big-O notation.
