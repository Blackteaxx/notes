## Polynomial Multiplication

(runtime just consider the number of flops)

- Input:
  - $A(x) = a_0 + a_1x + \cdots + a_{n-1}x^{n-1}$,
  - $B(x) = b_0 + b_1x + \cdots + b_{n-1}x^{n-1}$
- Output:
  - the vector of coefficients of $C(x) = A(x) \cdot B(x)$
  - $C(x) = c_0 + \cdots + c_{2n-2}x^{2n-2}$

$$
c_k = \sum_{i=0}^{k} a_i \cdot b_{k-i}
$$

pad the coefficient vector of $A(x), B(x)$ with 0 to the length of $2n-2$

- Extend: Integer Multiplication
  - Input: $A, B$, can be seen as the coefficient vector of $A(x), B(x)$ when the x is **10**
  - Distinction between the two problems: Integer Multiplication need to **consider the carry**

### 1. Naive Method

- Two for-loops to calculate all the coefficients of $C(x)$
- Runtime Analysis: $\Theta(n^2)$

  $$
    \frac{n}{4}^2 \leq T(n) \leq n^2
  $$

### 2. Karatsuba Algorithm

$$
A(x) = A_h(x) \cdot x^{\frac{n}{2}} + A_l(x) \\
B(x) = B_h(x) \cdot x^{\frac{n}{2}} + B_l(x)
$$

$$
A(x) \cdot B(x) = A_h(x) \cdot B_h(x) \cdot x^n + (A_h(x) \cdot B_l(x) + A_l(x) \cdot B_h(x)) \cdot x^{\frac{n}{2}} + A_l(x) \cdot B_l(x)
$$

- Runtime: $\mathcal{O}(n^{\log_2 3})$

### 3. Interpolation

Any polynomial of degree $n-1$ can **be uniquely determined by $n$ points**

**Proof**:

We have $n$ distinct points $(x_0, y_0), (x_1, y_1), \cdots, (x_{n-1}, y_{n-1})$, and we can construct a polynomial matrix which is called **Vandermonde Matrix**:

$$
\begin{bmatrix}
1 & x_0 & x_0^2 & \cdots & x_0^{n-1} \\
1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n-1} & x_{n-1}^2 & \cdots & x_{n-1}^{n-1}
\end{bmatrix}
\begin{bmatrix}
c_0 \\
c_1 \\
\vdots \\
c_{n-1}
\end{bmatrix} = \begin{bmatrix}
y_0 \\
y_1 \\
\vdots \\
y_{n-1}
\end{bmatrix}
$$

and we can get the solution of the polynomial by solving the linear system

According the result of the interpolation, we can get the polynomial by the points, and

$$
C(x) = A(x) \cdot B(x)
$$

We can list $2n-2$ diffenent points, and then we can get the polynomial by the interpolation

and the runtime of get one point is $\mathcal{O}(n)$, the sum of phase is $\mathcal{O}(n^2)$

and the runtime of get inverse is $\mathcal{O}(n^3)$

### 4. Fast Fourier Transform

- Types:

  - Fast Fourier Transform (FFT) is an algothrim
  - Discrete Fourier Transform (DFT) is a mathematical transform(matrix)

- DFT:

  $$
    F_{ij} = \omega^{ij}\\
    \omega = e^{\frac{-2\pi \sqrt{-1}}{n}}
  $$

- Complex Numbers
  - $z = a + bi$
  - $z = r\cdot e^{\theta i}$

If $n = 8$, we can get 8 points, namely we can represent the $8$ points using $\omega$ in complex plane

![img](https://img2023.cnblogs.com/blog/3436855/202407/3436855-20240718203819542-330989632.png)

$$
P(z) = p_0 + p_1z + p_2z + \cdots = \\
(p_0 + p_2z^2 + p_4z^4 + \cdots) + z(p_1 + p_3z^2 + \cdots) = \\
P_{\text{even}}(z^2) + zP_{\text{odd}}(z^2)
$$

and now we need to calculate the $P_{\text{even}}(\omega^2), P_{\text{odd}}(\omega^2)$

but when we square all the $n \ \omega$, we can get $\frac{n}{2}$ different points

$$
T(n) = 2T(\frac{n}{2}) + \mathcal{O}(n) = \Theta(n \log n)
$$

thus we can get the result with $\Theta(n \log n)$ given coefficients

1. Compute the DFT of $A(x), B(x)$ to get $A(\omega), B(\omega)$
2. Compute the point-wise product of $A(\omega), B(\omega)$ to get $C(\omega)$
3. return $F^{-1} \cdot C(\omega)$ to get the coefficients of $C(x)$
   - $F^{-1}$ is the inverse DFT matrix
   - $F^{-1} = \frac{1}{n} \cdot \bar{F}$
   - $\bar{A}v = \bar{A\bar{v}}$

