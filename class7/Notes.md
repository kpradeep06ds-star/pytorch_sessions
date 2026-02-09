---

# Gradient Descent in Linear Regression: From Scalars to Vectors
(Don't worry about bullet 7, we will discuss chain rule little later)
---

## 1. Scalar Linear Regression (Single Feature)

Consider a scalar regression model:

$$
[
\hat y = w_0 + w_1 x
]
$$

with mean squared error loss:

$$
[
\ell = \frac{1}{n} (y - \hat y)^2
]
$$

The gradients are:

$$
[
\frac{\partial \ell}{\partial w_0} = -\frac{2}{n}(y - \hat y)
]
$$


$$
[
\frac{\partial \ell}{\partial w_1} = -\frac{2}{n}(y - \hat y)x
]
$$

All quantities here are **scalars**.
Multiplication is commutative, and ordering does not matter.

---

## 2. Vector Linear Regression (Multiple Features)

Let:

* $(n)$ = number of samples
* $(d)$ = number of features

Define:

* Design matrix: $(X \in \mathbb{R}^{n \times d})$
* Parameter vector: $(w \in \mathbb{R}^{d})$
* Target vector: $(y \in \mathbb{R}^{n})$

Predictions:

$$
[
\hat y = Xw \in \mathbb{R}^n
]
$$

Residual (error):

$$
[
r = y - \hat y = y - Xw \in \mathbb{R}^n
]
$$

---

## 3. The Two Vector Spaces Involved

There are **two distinct vector spaces** in this problem:

### 1) Parameter space

$$
[
\mathbb{R}^d
]
$$


* Coordinates indexed by features
* Parameters (w) live here
* Gradients with respect to parameters must live here

### 2) Observation (data) space

$$
[
\mathbb{R}^n
]
$$

* Coordinates indexed by samples
* Targets $(y)$, predictions $(\hat y)$, and residuals $(r)$ live here

---

## 4. Loss Function as a Quadratic Form

The mean squared error loss is:

$$
[
\ell(w) = \frac{1}{n} |y - Xw|^2
]
$$

Written explicitly:

$$
[
\ell(w) = \frac{1}{n}(y - Xw)^T (y - Xw)
]
$$

This form is essential:

* The loss is a **scalar**
* Scalars arise from inner products
* Inner products require transposes

---

## 5. Why the Transpose Appears

The design matrix defines a linear map:

$$
[
X : \mathbb{R}^d \rightarrow \mathbb{R}^n
]
$$

* Forward direction: parameters → predictions
* Residuals live in (\mathbb{R}^n)

To compute gradients, information must be transported **back to parameter space**.

The transpose provides the adjoint map:

$$
[
X^T : \mathbb{R}^n \rightarrow \mathbb{R}^d
]
$$

This is the only linear map that converts residual information into parameter-space directions.

---

## 6. Gradient Computation (Correct Form)

The gradient of the loss with respect to (w) is:

$$
[
\nabla_w \ell
= -\frac{2}{n} X^T (y - Xw)
= -\frac{2}{n} X^T r
]
$$

Key observations:

* $(r \in \mathbb{R}^n)$
* $(X^T r \in \mathbb{R}^d)$
* The gradient lives in **parameter space**, as required

---

## 7. Why Order Matters in Vector Algebra

In scalar algebra:

$$
[
xr = rx
]
$$

In vector algebra:

$$
[
X^T r \neq r^T X
]
$$

* $(X^T r)$ is a vector in $(\mathbb{R}^d)$
* $(r^T X)$ is a row vector (dual representation)

Gradient descent requires a **direction in parameter space**, not a scalar or row vector.

---

## 8. Chain Rule Interpretation

The computation follows the chain:

$$
[
w \xrightarrow{X} \hat y \xrightarrow{y - \cdot} r \xrightarrow{|\cdot|^2} \ell
]
$$

Applying the chain rule:

$$
\begin{aligned}
\frac{d\ell}{dw}
&=
\frac{d\ell}{dr}
\cdot
\frac{dr}{d\hat y}
\cdot
\frac{d\hat y}{dw}
\end{aligned}
$$


* $(\frac{d\hat y}{dw} = X)$
* Backpropagation uses the adjoint: $(X^T)$

This explains the appearance of the transpose without invoking tricks.

---

## 9. Gradient Descent Update Rule

Using learning rate $(\eta)$:

$$
[
w \leftarrow w - \eta \nabla_w \ell
= w + \frac{2\eta}{n} X^T (y - Xw)
]
$$

Interpretation:

* Measure error in observation space
* Map it back to parameter space
* Update parameters in the direction that reduces loss

---

## 10. Connection to Least Squares and Geometry

Setting the gradient to zero:

$$
[
X^T (y - Xw) = 0
]
$$

This implies:

* Residual is orthogonal to the column space of (X)
* The solution is a **projection** of (y) onto the column space of (X)

This is the geometric meaning of least squares.

---

## 11. Why the “Bias as (x = 1)” Trick Feels Insufficient

In scalar regression:

* All spaces collapse into (\mathbb{R})
* Transposes are invisible

In vector regression:

* Spaces are distinct
* Transposes encode directionality
* The trick works algebraically but hides geometry

---

## 12. Final Summary

* Residuals live in observation space $(\mathbb{R}^n)$ (also known as data space and loosely sometimes sample space)
* Parameters live in parameter space $(\mathbb{R}^d)$
* (X) maps parameters to predictions
* (X^T) maps residuals back to parameters
* Gradients must live in parameter space
* Transposes are not tricks; they are adjoints

---

## One-Sentence Insight

> Gradient descent moves parameters by pulling prediction errors back through the transpose of the data matrix.

---
