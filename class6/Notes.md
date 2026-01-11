
It explains the **CNN architecture used in the notebook**, and **how parameters and dimensions are computed using the class defined in code**, in a way that reinforces *mechanical reasoning*, not memorization.


---

# CNN Architecture: Dimensions and Parameter Calculation

---

## 1. Purpose of Defining CNNs as Classes

In PyTorch, a convolutional neural network is typically defined by subclassing `nn.Module`.
This allows the architecture to be expressed as a **composition of layers**, each with a well-defined input–output transformation.

The class structure serves two purposes:

1. **Architectural clarity**
   Each layer explicitly defines how data flows through the network.

2. **Deterministic shape and parameter computation**
   All spatial dimensions and parameter counts can be computed *before training begins*.

---

## 2. Canonical CNN Architecture Used in the Code

The CNN architecture discussed in class follows this general pattern:

```
Input
 → Conv2d
 → BatchNorm2d
 → Activation
 → Pooling
 → (repeat)
 → Flatten
 → Linear
 → Output
```

Each block modifies:

* **Spatial dimensions** (H, W)
* **Channel depth** (C)
* **Parameter count**

Understanding these changes is essential for debugging and model design.

---

## 3. Input Tensor Convention

For image data, the input tensor shape is:

```
(B, C, H, W)
```

Where:

* `B` = batch size
* `C` = number of channels
* `H` = height
* `W` = width

All subsequent dimension calculations operate on `(C, H, W)`.

---

## 4. Convolution Layer: Dimension Calculation

For a convolution layer defined as:

```python
nn.Conv2d(
    in_channels = C_in,
    out_channels = C_out,
    kernel_size = K,
    stride = S,
    padding = P,
    dilation = D
)
```

### Output spatial dimensions

$$
[
H_{\text{out}} =
\left\lfloor
\frac{H + 2P - D(K - 1) - 1}{S}
\right\rfloor + 1
]
$$


$$
[
W_{\text{out}} =
\left\lfloor
\frac{W + 2P - D(K - 1) - 1}{S}
\right\rfloor + 1
]
$$

### Output shape

```
(B, C_out, H_out, W_out)
```

---

## 5. Convolution Layer: Parameter Count

### Weight parameters

Each output channel has a kernel of shape:

```
(C_in, K, K)
```

Total weights:

$$
[
C_{\text{out}} \times C_{\text{in}} \times K \times K
]
$$

### Bias parameters

$$
[
C_{\text{out}}
]
$$

### Total parameters

$$
[
\boxed{
C_{\text{out}} \times C_{\text{in}} \times K \times K
;+;
C_{\text{out}}
}
]
$$

---

## 6. Batch Normalization: Parameters and Dimensions

For:

```python
nn.BatchNorm2d(C)
```

### Parameters per channel

* Learnable scale $ (`γ`) $
* Learnable shift $ (`β`) $

### Total learnable parameters

$$
[
\boxed{2 \times C}
]
$$

### Important clarification

Running mean and variance:

* Are **buffers**, not parameters
* Do not contribute to gradient updates

BatchNorm **does not change tensor shape**.

---

## 7. Pooling Layers: Dimension Changes (No Parameters)

Pooling layers reduce spatial resolution but introduce **no learnable parameters**.

### Example: MaxPool2d

```python
nn.MaxPool2d(kernel_size=K, stride=S)
```

Output dimensions:

$$
[
H_{\text{out}} =
\left\lfloor
\frac{H - K}{S}
\right\rfloor + 1
]
$$


$$
[
W_{\text{out}} =
\left\lfloor
\frac{W - K}{S}
\right\rfloor + 1
]
$$

Channels remain unchanged.

---

## 8. Flattening Before Fully Connected Layers

Before passing data into a `Linear` layer, the tensor is flattened:

```
(B, C, H, W) → (B, C × H × W)
```

This step is purely a **reshape operation** and has no parameters.

---

## 9. Fully Connected (Linear) Layer: Parameters

For:

```python
nn.Linear(in_features = N, out_features = M)
```

### Parameter count

$$
[
\boxed{
N \times M ;+; M
}
]
$$

Where:

* `N` = flattened input size
* `M` = number of output neurons

---

## 10. Using the Class to Validate Dimensions

The CNN class defined in the code allows:

* Step-by-step verification of tensor shapes
* Manual confirmation of:

  * Convolution output sizes
  * Pooling reductions
  * Flattened feature dimensions
* Early detection of shape mismatches before training

This reinforces the principle:

> **CNN architectures are deterministic mathematical objects, not trial-and-error constructs**

---

## 11. End-to-End Example Reasoning

Given:

* Input: `(1, 28, 28)`
* Conv2d: `1 → 8`, kernel `3×3`, padding `1`
* Pooling: `2×2`
* Flatten
* Linear: `N → 10`

One can compute:

1. Output spatial size after convolution
2. Reduced size after pooling
3. Flattened dimension
4. Linear layer parameter count

This entire process can be validated **without executing the model**.

---

## 12. Conceptual Summary

* CNN architecture design is governed by **exact arithmetic**
* Every layer has predictable effects on:

  * Shape
  * Parameter count
* Class-based definitions enable:

  * Modular reasoning
  * Debuggable design
  * Scalable architectures


---

# Loss Functions, Likelihood, and Data Distributions

---

## 1. What a Loss Function Really Is

A **loss function** is a scalar-valued function that measures how well a model’s predictions agree with observed data.

Formally, a loss function defines an **objective** that training attempts to minimize.

$$
[
\text{Training} \quad \Longleftrightarrow \quad \text{Optimization of an objective function}
]
$$

Loss functions are therefore not specific to:

* Neural networks
* Supervised learning
* Classification problems

They are a general concept in optimization.

---

## 2. Loss Functions as Negative Log-Likelihood

### Central principle

Most commonly used loss functions in machine learning arise from **Maximum Likelihood Estimation (MLE)**.

Instead of asking:

> “How wrong is the prediction?”

MLE asks:

> “How likely is the observed data under the model?”

Training then becomes:

$$
[
\text{maximize } p(y \mid x, \theta)
\quad \Longleftrightarrow \quad
\text{minimize } -\log p(y \mid x, \theta)
]
$$



---

### Why log-likelihood is used

* Converts products into sums
* Improves numerical stability
* Simplifies gradients
* Preserves optima (log is monotonic)

Thus, **loss = negative log-likelihood** is not arbitrary — it is mathematically motivated.

---

## 3. Conditional Distributions and Model Assumptions

A model does not directly predict a target value.
Instead, it models a **conditional probability distribution**:

$$
[
p(y \mid x)
]
$$

The choice of loss function is therefore equivalent to choosing a **conditional distribution** for the target.

This assumption is fundamental and often implicit.

---

## 4. Regression Losses and Their Distributions

### 4.1 Gaussian Distribution → Mean Squared Error (MSE)

Assumption:

$$
[
y \mid x \sim \mathcal{N}(\mu(x), \sigma^2)
]
$$

Negative log-likelihood (ignoring constants):

$$
[
\mathcal{L}_{\text{MSE}} = (y - \mu(x))^2
]
$$

#### Interpretation

* Errors are symmetric
* Large errors are penalized heavily
* Sensitive to outliers

---

### 4.2 Laplace Distribution → Mean Absolute Error (MAE)

Assumption:

$$
[
y \mid x \sim \text{Laplace}(\mu(x), b)
]
$$

Negative log-likelihood:

$$
[
\mathcal{L}_{\text{MAE}} = |y - \mu(x)|
]
$$

#### Interpretation

* Heavier tails than Gaussian
* More robust to outliers
* Penalizes deviations linearly

---

### Key takeaway (Regression)

> Choosing between MSE and MAE is not cosmetic — it reflects an assumption about the **noise distribution** in the data.

---

## 5. Binary Classification and Bernoulli Distribution

### 5.1 Bernoulli Assumption

For binary targets:

$$
[
y \in {0,1}
]
$$

Assume:

$$
[
y \mid x \sim \text{Bernoulli}(p(x))
]
$$

Where:

$( p(x) )$ is the model’s predicted probability

---

### 5.2 Binary Cross-Entropy Loss (BCE)

Negative log-likelihood of Bernoulli distribution:

$$
[
\mathcal{L}_{\text{BCE}} = -\left[y \log p + (1-y)\log(1-p)\right]
]
$$

This is **not heuristically designed** — it follows directly from likelihood theory.

---

### Practical PyTorch distinction

* `BCELoss`: expects probabilities (after sigmoid)
* `BCEWithLogitsLoss`: expects raw logits
  (numerically stable, preferred)

---

## 6. Multiclass Classification and Categorical Distribution

### 6.1 Categorical Distribution

For ( K ) classes:

$$
[
y \mid x \sim \text{Categorical}(p_1, p_2, \dots, p_K)
]
$$

---

### 6.2 Cross-Entropy Loss

Negative log-likelihood of categorical distribution:

$$
[
\mathcal{L}*{\text{CE}} = -\log p*{y}
]
$$

In PyTorch:

* `CrossEntropyLoss` internally applies:

  * `LogSoftmax`
  * `NLLLoss`

Thus:

> **Do not apply Softmax before `CrossEntropyLoss`**

---

## 7. Unified View of Loss Functions

| Task                      | Distribution Assumed | Loss Function |
| ------------------------- | -------------------- | ------------- |
| Regression                | Gaussian             | MSE           |
| Robust Regression         | Laplace              | MAE           |
| Binary Classification     | Bernoulli            | BCE           |
| Multiclass Classification | Categorical          | Cross Entropy |

---

## 8. Important Conceptual Clarification

Loss functions are **not tied to labels**.

Examples:

* PCA minimizes reconstruction error
* K-Means minimizes within-cluster variance
* Autoencoders minimize reconstruction loss

Therefore:

> **Loss functions are objective functions, not supervision-specific tools**

---

## 9. Why “Choosing the Right Loss” Matters

The loss function encodes:

* Assumptions about noise
* Sensitivity to outliers
* Confidence calibration
* Optimization behavior

Incorrect assumptions lead to:

* Poor convergence
* Biased predictions
* Unstable gradients

---

## 10. Conceptual Summary

1. Models learn conditional distributions, not point values
2. Loss functions arise from negative log-likelihood
3. Different losses correspond to different distributional assumptions
4. Optimization is the process of fitting these distributions to data


> **Loss defines the landscape, optimizers define how we move on it**

Loss functions are not “choices from an API”.
They are **mathematical statements about how the world generates data**.

Understanding this removes memorization and replaces it with reasoning.

---

## 11. Next Class (As Concluded in Session)

### Topic: Optimizers and Learning Rate Schedulers

Planned coverage:

* Gradient descent variants
* Momentum and adaptive methods
* Why different optimizers exist
* Learning rate schedules:

  * Step
  * Cosine
  * Warmup
* Optimizer–loss interaction


---



