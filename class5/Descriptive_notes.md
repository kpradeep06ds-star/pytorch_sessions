
---

# PyTorch Class Notes — Callable Objects, `nn.Module`, BatchNorm, and Convolution Mechanics

---

## 1. Callable Objects in Python (`__call__`)

In Python, an object can behave like a function if the special method `__call__()` is defined.
When such an object is invoked using parentheses, Python internally executes its `__call__()` method.

This concept is essential for understanding PyTorch’s execution model.

Example intuition:

* An object is created from a class.
* Calling the object (`obj(x)`) triggers `obj.__call__(x)`.
* This mechanism allows objects to encapsulate both **state** and **behavior**.

This property is later used by PyTorch to make neural-network modules executable like functions.

---

## 2. `forward()` vs `__call__()` in PyTorch

### Core principle

In PyTorch, **we never call `forward()` directly**.
Instead, we call the module itself:

```python
y = model(x)
```

This works because `nn.Module` implements `__call__()` internally.

---

### Execution flow inside `nn.Module`

When `model(x)` is executed:

1. `nn.Module.__call__()` is invoked
2. PyTorch performs internal bookkeeping:

   * Autograd graph construction
   * Forward and backward hooks
   * Training vs evaluation mode handling
3. `forward(x)` is finally called internally
4. The output tensor is returned

Therefore:

> **We override `forward()`, but PyTorch owns `__call__()`**

This separation is deliberate and fundamental to PyTorch’s design.

---

### Correct mental model

* `forward()` defines **what computation is performed**
* `__call__()` defines **how and when the computation is executed**

Calling `forward()` directly bypasses PyTorch’s machinery and must be avoided.

---

## 3. Object-Oriented Design Patterns Used in PyTorch

### Inheritance

Used when defining new modules:

```python
class MyModel(nn.Module):
    ...
```

This establishes an *is-a* relationship.

---

### Composition (dominant pattern in PyTorch)

Modules are constructed by **owning layers**, not inheriting them:

```python
self.conv = nn.Conv2d(...)
self.bn   = nn.BatchNorm2d(...)
```

This expresses a *has-a* relationship and enables modular, reusable design.

---

### Important clarification

PyTorch models are **not built by stacking inheritance**, but by composing layers inside a module.
This avoids tight coupling and keeps architectures flexible.

---

## 4. Batch Normalization (Concept and Implementation)

### Purpose

Batch Normalization stabilizes and accelerates training by normalizing activations **per channel**.

In CNNs, BatchNorm operates independently on each feature map.

---

### Input structure

For 2D BatchNorm, the expected input shape is:

```
(B, C, H, W)
```

Normalization is performed over:

* Batch dimension
* Spatial dimensions `(H, W)`
* Separately for each channel `C`

---

### Mathematical operation (per channel)

Given a channel tensor ( x ):

$$
[
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
]
$$

where:

* $( \mu )$ is the mean
* $( \sigma^2 )$ is the variance
* $( \epsilon )$ is a numerical stability constant

---

### Important correction / clarification

Using:

```python
std(unbiased=False)
```

is **correct** in this context.

Deep-learning frameworks typically use **population variance**, not sample variance, for normalization layers. This is intentional and aligns with PyTorch’s internal BatchNorm behavior.

---

### Shift and scale intuition

Adding a constant to the input:

* Changes the mean
* Does **not** change the standard deviation

This demonstrates that BatchNorm explicitly removes mean shifts while standardizing scale.

---

## 5. Standard CNN Block Ordering (Clarified)

A common and effective CNN block ordering is:

```
Convolution → BatchNorm → Activation → (Optional) Pooling
```

### Rationale

* **Convolution** extracts features
* **BatchNorm** stabilizes distributions before nonlinearity
* **Activation (ReLU / variants)** introduces nonlinearity
* **Pooling** provides spatial downsampling and invariance

---

### Clarification regarding Dropout

In CNNs, dropout is less commonly applied at the pixel level because:

* Neighboring pixels are highly correlated
* Dropping individual pixels has limited effect

Dropout is more effective:

* At channel level
* In fully connected layers
* Or via structured variants (e.g., Dropout2d)

---

## 6. Convolution as Sum-Product (Dot Product over Patches)

### Fundamental definition

A **dot product** is a *sum of products*.

A **2D convolution** is a repeated dot product between:

* A flattened local input patch
* A flattened kernel

---

### One output pixel computation

For a given output channel:

1. Extract a local patch:

   ```
   (C × K × K)
   ```
2. Multiply element-wise with kernel weights
3. Sum all products
4. Add bias

Mathematically:

$$
[
y = \sum_{c,h,w} x_{c,h,w} \cdot w_{c,h,w} + b
]
$$

---

### Equivalent computation forms

All the following are mathematically identical:

* Element-wise multiply + sum
* Flatten + dot product
* Einstein summation (`einsum`)
* Matrix multiplication after unfolding

---

### Key insight

> **Convolution is not mysterious**
> It is structured linear algebra applied repeatedly across space.

---

## 7. Role of `einsum`

Einstein summation notation is not required for simple cases, but it becomes extremely powerful when:

* Tensor dimensions grow
* Operations involve multiple shared axes
* Clarity is preferred over nested loops

Its inclusion builds long-term tensor fluency.

---

## 8. Conceptual Closure of the Class

The following equivalence is essential:

* Dot product → sum-product
* Convolution → repeated sum-product over spatial locations
* CNNs → hierarchical composition of these operations

This understanding removes the “black box” nature of convolution layers.

---

## 9. Homework (As Stated in Notebook)

**Implement from scratch (class-based):**

* Max Pooling
* Average Pooling
* Dropout

The implementation should follow the same style used for:

* Convolution
* Batch Normalization

---
# A Note about regularisation


## Regularization in Deep Learning: A Unified View

Regularization refers to a collection of techniques used to **control model complexity**, improve **generalization**, and reduce **overfitting**.
Although textbooks often present regularization methods as a flat list, they can be meaningfully grouped based on *where* and *how* they act in the learning pipeline.

This section organizes regularization into **three conceptual categories**:

1. **Data-based regularization**
2. **Algorithmic regularization**
3. **Process / Network-based regularization**

This classification is not standard terminology, but it provides a **useful mental model** for understanding why different techniques work.

---

## 1. Data-Based Regularization

### Definition

Data-based regularization modifies or augments the **training data distribution** to prevent the model from memorizing spurious patterns.

### Core idea

> Increase effective data diversity without changing the model.

---

### Common examples

* **Data augmentation**

  * Image flips, rotations, crops
  * Color jittering
  * Noise injection
* **Mixup / CutMix**

  * Linear or spatial mixing of samples
* **Input noise**

  * Gaussian noise added to inputs

---

### Effect on learning

* Forces the model to learn **invariances**
* Reduces sensitivity to small perturbations
* Acts before the model ever sees the data

---

### Key insight

Data-based regularization operates **outside the model**.
It reshapes the learning problem rather than constraining the parameters directly.

---

## 2. Algorithmic Regularization

### Definition

Algorithmic regularization alters the **optimization objective or optimization dynamics** to penalize overly complex solutions.

### Core idea

> Restrict what the optimizer is allowed to prefer.

---

### Common examples

* **L2 regularization (weight decay)**
$$
  [
  \mathcal{L} = \mathcal{L}_{data} + \lambda |\theta|^2
  ]
$$

* **L1 regularization**

  * Encourages sparsity
* **Early stopping**

  * Stops optimization before overfitting
* **Label smoothing**

  * Softens hard targets
* **Gradient clipping**

  * Controls optimization instability

---

### Important clarification

Although often grouped together, these methods regularize **different things**:

* Weight decay regularizes **parameter magnitude**
* Early stopping regularizes **training time**
* Label smoothing regularizes **target certainty**

---

### Key insight

Algorithmic regularization works by **changing the loss landscape or optimization trajectory**, not the model structure.

---

## 3. Process / Network-Based Regularization

### Definition

Process-based (or network-based) regularization modifies the **internal computation graph or architecture** during training.

### Core idea

> Introduce controlled uncertainty inside the model.

---

### Common examples

* **Dropout / Dropout2d**
* **Stochastic depth**
* **Batch Normalization**
* **Layer Normalization**
* **Residual connections (indirectly)**

---

### Important correction / clarification

Batch Normalization is **not designed primarily as a regularizer**, but:

* It introduces noise due to batch statistics
* This noise has a **regularizing side effect**

Therefore, BatchNorm is best viewed as:

> *An optimization technique with regularization consequences*

---

### Why dropout behaves differently in CNNs

* Dropping individual pixels is ineffective due to spatial correlation
* Channel-wise dropout (`Dropout2d`) is more meaningful
* In modern CNNs, dropout is often reduced or removed in favor of:

  * Data augmentation
  * BatchNorm
  * Weight decay

---

### Key insight

Process-based regularization operates **during forward passes**, not at the data or loss level.

---

## 4. Relationship Between the Three Forms

| Category      | Acts On             | When It Operates    |
| ------------- | ------------------- | ------------------- |
| Data-based    | Inputs              | Before training     |
| Algorithmic   | Loss / optimizer    | During optimization |
| Process-based | Network computation | During forward pass |

---

## 5. Why This Classification Is Useful

Traditional presentations list techniques without explaining **why they belong together**.
This three-part classification answers:

* *What is being constrained?*
* *Where is randomness introduced?*
* *When does regularization act?*

It also explains why modern pipelines often rely more on:

* Data augmentation
* Weight decay
* Normalization layers

…and less on classical dropout-heavy designs.

---

## 6. Unifying Principle

> All regularization methods limit a model’s ability to fit noise,
> but they do so at **different stages of the learning pipeline**.

Understanding *where* a method acts is more important than memorizing *which* method to use.

---

## Bridge to Next Class (Loss Functions)

This classification naturally prepares the ground for loss functions:

* Losses themselves often include **regularization terms**
* Many unsupervised methods (PCA, K-Means) are entirely defined by their objective functions
* Regularization is inseparable from optimization

The transition from **regularization → loss functions** is therefore conceptually continuous, not a topic switch.


---

## 10. Next Class (From Final Notebook Cell)

### Topics to be covered

1. **Loss functions**

   * Regression losses
   * Classification losses

2. **Why PyTorch provides many variants of the same loss**

   * Numerical stability
   * Input expectations (logits vs probabilities)
   * Gradient behavior

3. **Clarifying a common misconception**

   * Losses are not exclusive to supervised learning
   * PCA, K-Means, and many unsupervised methods also optimize objective functions

4. **Unifying idea**

   > **Losses are everywhere — they are the building blocks of learning**

---

## Final Conceptual Summary

This class establishes three pillars:

1. **Execution mechanics** (`__call__` vs `forward`)
2. **Stabilization mechanisms** (BatchNorm)
3. **Computational foundations** (convolution as dot product)

With this foundation, moving into loss functions becomes natural, not abrupt.

---

