# Feature Scaling — Normalization


---

## What is Normalization?

**Normalization** is a technique often applied as part of **data preparation for machine learning**.

> The goal of normalization is to **change the values of numeric columns** in the dataset to use a **common scale**, without distorting differences in the ranges of values or losing information.

In simple words: make all features live in the **same numerical neighborhood** so no single feature dominates the model.

---

## Types of Normalization

Feature Scaling has two main branches. Normalization sits on the right:

```
                        Feature Scaling
                       /               \
           Standardization          Normalization
           (Z-score, μ=0, σ=1)     /      |       \
                              MinMax   MaxAbs   Robust
                              Scaling  Scaling  Scaling
                                  \
                               Mean Normalization
```

| Technique | Output Range | Best For |
|---|---|---|
| **MinMax Scaling** | `[0, 1]` | Most use cases, image pixel data |
| **Mean Normalization** | `[-1, 1]` | Centered, bounded data |
| **Max Absolute Scaling** | `[-1, 1]` | Sparse data (keeps zeros as zeros) |
| **Robust Scaling** | Varies | Data with many **outliers** |

---

## 1. MinMax Scaling

### Intuition
MinMax Scaling **squeezes** all values into the range `[0, 1]`.
- The **minimum** value in a column becomes -> `0`
- The **maximum** value in a column becomes -> `1`
- Everything else falls **proportionally** in between

### Formula

$$X'_i = \frac{X_i - X_{min}}{X_{max} - X_{min}}$$

```
         X_i  -  X_min
X_i'  =  ──────────────────
         X_max  -  X_min
```

| Variable | Meaning |
|---|---|
| `X_i` | The original data point to scale |
| `X_min` | The minimum value in that column |
| `X_max` | The maximum value in that column |
| `X_i'` | The new scaled value (always between 0 and 1) |

---

### Step-by-Step Example (Weight Column)

Suppose we have a `Weight` column from the dataset:

| Row | Weight (kg) |
|-----|------------|
| 1   | 130 (max)  |
| 2   | 67         |
| 3   | 81         |
| 4   | 61         |
| 5   | 32 (min)   |
| 6   | 54         |

- **X_min = 32**, **X_max = 130**

**Scaling the Maximum value (130 kg):**
```
X' = (130 - 32) / (130 - 32)
   =    98      /     98
   =    1.0
```

**Scaling the Minimum value (32 kg):**
```
X' = (32 - 32) / (130 - 32)
   =     0     /     98
   =    0.0
```

**Scaling an intermediate value (67 kg):**
```
X' = (67 - 32) / (130 - 32)
   =    35     /     98
   =   0.357
```

**After scaling — the transformed Weight column:**

| Row | Weight (kg) | Weight' (scaled) |
|-----|------------|-----------------|
| 1   | 130        | **1.000**       |
| 2   | 67         | **0.357**       |
| 3   | 81         | **0.500**       |
| 4   | 61         | **0.296**       |
| 5   | 32         | **0.000**       |
| 6   | 54         | **0.224**       |

> All values are now squeezed into a clean `[0, 1]` range!

---

## 2. Visualizing MinMax Scaling

The key transformation: the data cloud **moves** and **squishes** into a perfectly bounded `[0, 1] x [0, 1]` box.

```
  BEFORE MinMax Scaling                     AFTER MinMax Scaling
  (Original units: kg, cm, etc.)            (All values between 0 and 1)

  Height                                    Height'
    │                                       1.0 ┼ ─ ─ ─ ─ ─ ─ ─ ─┐
 180│         ●  ●                              │   ● ●  ●       │
    │       ●  ●  ●                          0.5┼     ● ●  ●     │
 160│     ●   ●  ●                              │  ●    ●  ●     │
    │       ●  ●                            0.0 ┼ ─ ─ ─ ─ ─ ─  ─ ┘
 140│                                           └──┼─────────────┼── Weight'
    └─────────────────── Weight              (0,0) 0             1
       40   60   80  100
```

### Key Takeaways

| Property | MinMax Scaling | Standardization |
|---|---|---|
| **Output Range** | Always `[0, 1]` | No fixed range (can be -∞ to +∞) |
| **Mean after scaling** | Not necessarily 0 | Always **0** |
| **Affected by outliers?** | **YES** — heavily | Less affected |
| **Shape of data preserved?** | Yes | Yes |
| **Zeros preserved?** | No | No |

---

## Impact of Outliers on MinMax Scaling

This is the **biggest weakness** of MinMax Scaling.

**Example:** Suppose one person in the dataset weighs **500 kg** (an outlier).
- $X_{min} = 32$, $X_{max} = 500$

Now a normal person weighing 67 kg gets scaled to:
```
X' = (67 - 32) / (500 - 32) = 35 / 468 = 0.074
```

> All "normal" data gets **crushed** into the tiny range `[0.0, 0.1]`, making them nearly indistinguishable. This is why **Robust Scaling** was invented — it is immune to outliers!

---

## Code (Scikit-Learn)

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('wine_data.csv')

# Separate features and target
X = df.drop('Class label', axis=1)
y = df['Class label']

# Apply MinMax Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame for inspection
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
print(X_scaled_df.head())
```

**Before Scaling (sample):**

| Alcohol | Malic acid | Ash |
|---------|------------|-----|
| 14.23   | 1.71       | 2.43 |
| 13.20   | 1.78       | 2.14 |
| 13.16   | 2.36       | 2.67 |

**After MinMax Scaling:**

| Alcohol | Malic acid | Ash |
|---------|------------|-----|
| 1.000   | 0.107      | 0.489 |
| 0.708   | 0.121      | 0.340 |
| 0.695   | 0.246      | 0.574 |

> Notice how `Alcohol = 14.23` (the max) maps to exactly `1.000`!

---

## Visualizing the Wine Dataset (Before vs After)

```python
import seaborn as sns
import matplotlib.pyplot as plt

color_dict = {1: 'red', 3: 'green', 2: 'blue'}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before Scaling
sns.scatterplot(
    x=df['Alcohol'], y=df['Malic acid'],
    hue=df['Class label'], palette=color_dict, ax=axes[0]
)
axes[0].set_title('Before MinMax Scaling')

# After Scaling
sns.scatterplot(
    x=X_scaled_df['Alcohol'], y=X_scaled_df['Malic acid'],
    hue=y, palette=color_dict, ax=axes[1]
)
axes[1].set_title('After MinMax Scaling')

plt.tight_layout()
plt.show()
```

---

## Normalization vs Standardization — When to Use?

| Situation | Use |
|---|---|
| Data has **no significant outliers** | MinMax Normalization |
| You need values strictly between `0` and `1` | MinMax Normalization |
| You are using a **Neural Network** or **image data** | MinMax Normalization |
| Data has **significant outliers** | Robust Scaling |
| You are using **KNN, K-Means, PCA, SVM** | Standardization (Z-score) |
| Data distribution is **Gaussian (Normal)** | Standardization (Z-score) |
