# Feature Engineering

> **Definition:** Feature engineering is the process of using **domain knowledge** to extract features from raw data. These features can be used to improve the performance of machine learning algorithms.

---

## Overview: The Feature Engineering Tree

Feature Engineering is broken down into four main branches:

```
                        Feature Engineering
                               |
       ┌───────────────────────┼───────────────────────┐
       │               │               │               │
Feature             Feature        Feature         Feature
Transformation     Construction    Scaling &      Extraction
       │                           Selection
       │
  ├── Missing Value Imputation
  ├── Handling Categorical Features
  ├── Outlier Detection
  └── Feature Scaling
```

## The ML Pipeline (Where Feature Engineering Fits)

```
Database → Data Preprocessing & Cleaning
                    ↓
        Feature Selection & Engineering
                    ↓
                Modelling
                    ↓
        Model Evaluation & Tuning
                    ↓
        Deployment & Monitoring
                    ↑
     (Iterate till satisfactory model performance)
```

---

## 1. Feature Transformation

### 1.1 Missing Values Imputation

When a dataset has missing values (`NaN`), we must **impute** (fill in) those values before feeding the data into a model (most ML algorithms cannot handle `NaN`).

**Example:** The average age is **26.0**. We use this to fill in the missing values.

**Before Imputation:**

| ID | City   | Age |
|----|--------|-----|
| 1  | Lisbon | 25  |
| 2  | Berlin | 25  |
| 3  | Lisbon | 30  |
| 4  | Berlin | 19  |
| 5  | Berlin | NaN | ← Missing!
| 7  | Berlin | NaN | ← Missing!
| 8  | Berlin | 25  |
| 9  | Berlin | 25  |
| 10 | Madrid | 25  |

**After Imputation (filled with Average_Age = 26.0):**

| ID | City   | Age | Married? |
|----|--------|-----|----------|
| 1  | Lisbon | 25  | 1        |
| 2  | Berlin | 25  | 0        |
| 3  | Lisbon | 30  | 1        |
| 4  | Berlin | 19  | 0        |
| 5  | Berlin | **26** | 1     | ← Filled!
| 6  | Berlin | 36  | 1        |
| 7  | Berlin | **26** | 1     | ← Filled!
| 9  | Berlin | 25  | 1        |
| 10 | Madrid | 25  | 1        |

---

### 1.2 Handling Categorical Values

ML models only understand **numbers**, not text like "Dog", "Cat", or "Yes/No". We must convert these text categories into numerical representations.

There are two main techniques:

**Technique A — Label Encoding:** Simply assign each category a number (0, 1, 2, 3...).
> ⚠️ Problem: The model may incorrectly assume that 3 (Horse) is "greater than" 0 (Dog), implying a false ranking!

**Technique B — One-Hot Encoding:** Create a separate binary (0/1) column for **each** category. This removes the false ranking problem.

**Before One-Hot Encoding (Original):**

| Index | Animal |
|-------|--------|
| 0     | Dog    |
| 1     | Cat    |
| 2     | Sheep  |
| 3     | Horse  |
| 4     | Lion   |

**After One-Hot Encoding → Each animal gets its own binary column:**

| Index | Dog | Cat | Sheep | Lion | Horse |
|-------|-----|-----|-------|------|-------|
| 0     | 1   | 0   | 0     | 0    | 0     |
| 1     | 0   | 1   | 0     | 0    | 0     |
| 2     | 0   | 0   | 1     | 0    | 0     |
| 3     | 0   | 0   | 0     | 0    | 1     |
| 4     | 0   | 0   | 0     | 1    | 0     |

> Each row now has a `1` in exactly the column that matches its animal, and `0` everywhere else. The model can now correctly treat all animals as equal, unranked categories!

> **Key Insight:** One-Hot Encoding increases the number of columns — a dataset with 1 column of 5 categories becomes 5 new columns. This is important to keep in mind!

---

### 1.3 Outlier Detection

Outliers are extreme data points that are far from the rest of the data. They can severely distort a model's predictions.

**Visual Comparison:**

| With Outliers | After Outliers Removed |
|---|---|
| The regression line is pulled toward the outlier, giving a poor fit for the majority of the data. | The regression line fits the data much better — **a much better fit!** |

```
With Outliers:                    Outliers Removed:
    fit                               fit
  ^                                 ^
  |    ●  ← outlier                 |      /
  |  ● ●  ●                         |   ● / ●
  |  ●  ●                           |  ● ●
  |●●                               | ●●
  └──────────> speed                └──────────> speed
```

> **Rule of Thumb:** An outlier can ruin a model that works perfectly for the other 999 data points. Always detect and handle them!

---

### 1.4 Feature Scaling

Different features have different units and ranges (e.g., Age: 27–48, Salary: 48,000–84,000). Models like KNN and SVM that use **distance calculations** will be biased toward the feature with the larger range.

**Example Dataset (Before Scaling):**

| Country | Age | Salary | Purchased |
|---------|-----|--------|-----------|
| France  | 44  | 72000  | Yes       |
| France  | 27  | 48000  | Yes       |
| Spain   | 30  | 84000  | Yes       |
| Spain   | 36  | 61000  | Yes       |
| Germany | 40  | 81000  | Yes       |
| France  | 48  | 52000  | Yes       |
| Spain   | 37  | 75000  | Yes       |

**The Problem:**
The Euclidean distance formula is:

```
distance = √( (x₁-x₂)² + (y₁-y₂)² )
```

If `y` is Salary (e.g., 32,000 vs 33,000), its squared difference **completely dominates** the calculation, making the Age column effectively invisible to the model!

**After Feature Scaling (Min-Max Normalization — range `-1 to +1`):**

| Country | Age (scaled) | Salary (scaled) | Purchased |
|---------|-------------|-----------------|----|
| France  | 0.81        | 0.58            | 1  |
| France  | -1.00       | -1.00           | 1  |
| Spain   | -0.60       | 1.00            | 1  |
| Spain   | 0.08        | -0.08           | 1  |
| Germany | 0.60        | 0.86            | 1  |
| France  | 1.00        | -0.73           | 1  |
| Spain   | 0.22        | 0.72            | 1  |

> Now `Age` and `Salary` are on the **same scale** (-1 to +1). Every feature contributes equally to the distance calculation — no single feature can dominate the model!

---

## 2. Feature Construction

Feature construction means creating **brand new columns** from the existing data using domain knowledge. These new features often help the model find patterns it couldn't see before.

**Before Feature Construction — Titanic Dataset (Original):**

| ID | Pclass | Name                                          | Sex    | Age | SibSp | Parch | Fare    | Embarked |
|----|--------|-----------------------------------------------|--------|-----|-------|-------|---------|----------|
| 1  | 3      | Braund, Mr. Owen Harris                       | male   | 23  | 1     | 0     | A/5 3.2 | S        |
| 2  | 1      | Cumings, Mrs. John Bradley (Florence Sing...) | female | 22  | 1     | 0     | 71.83   | C        |

**After Feature Construction — New engineered columns added:**

| ID | Pclass | Name                     | Sex    | Age | SibSp | Parch | Fare  | Embarked | **Title** | **Family_Size** | **Is_Alone** | **Age_Group** |
|----|--------|--------------------------|--------|-----|-------|-------|-------|----------|-----------|-----------------|--------------|---------------|
| 1  | 3      | Braund, Mr. Owen Harris  | male   | 23  | 1     | 0     | 3.2   | S        | **Mr**    | **2**           | **0**        | **Adult**     |
| 2  | 1      | Cumings, Mrs. John...    | female | 22  | 1     | 0     | 71.83 | C        | **Mrs**   | **2**           | **0**        | **Adult**     |

**How the new features were constructed:**

| New Feature   | How It Was Made                            | Why It's Useful |
|---------------|--------------------------------------------|-----------------|
| `Title`       | Extracted from `Name` (Mr., Mrs., Miss...) | Title reveals social status & age group |
| `Family_Size` | `SibSp + Parch + 1`                        | Larger families had different survival rates |
| `Is_Alone`    | `1` if `Family_Size == 1`, else `0`        | Solo travelers behaved differently |
| `Age_Group`   | Bins Age into Child/Teen/Adult/Senior      | Age categories capture non-linear patterns |

---

## 3. Feature Selection

Feature selection is the process of choosing **only the most relevant features** and removing redundant or irrelevant ones.

**Why?** Using too many features can cause:
- **Overfitting** — the model memorizes the training data instead of learning patterns.
- **Slow training** — more features = more computation.
- **The Curse of Dimensionality** — in high dimensions, data becomes very sparse.

**Visual Example — MNIST Digit Recognition:**

The MNIST dataset contains images of handwritten digits (0-9). Each image is 28×28 pixels = **784 features**. The vast majority of those pixels are just white background — they add noise and slow the model down.

**Before Feature Selection — All 784 pixel features:**
```python
train_id = pd.read_csv('train.csv')
print(train_id.shape)  # Output: (42000, 785)  → 784 pixel columns + 1 label column!
```

| label | pixel0 | pixel1 | pixel2 | pixel3 | ... | pixel783 |
|-------|--------|--------|--------|--------|-----|----------|
| 1     | 0      | 0      | 0      | 0      | ... | 0        |
| 0     | 0      | 0      | 0      | 0      | ... | 0        |

*(Most pixel columns are 0 — they are empty background pixels!)*

**After Feature Selection — Only the important, non-zero pixel features are kept:**

The center pixels of the 28×28 image (where the actual digit is drawn) are retained. The border pixels (always 0) are discarded. This reduces the dataset from 784 features to only the meaningful ones.

```
Before: 28×28 grid = 784 features     After: Only center pixels kept
□□□□□□□□□□□□□□□□□□□□□□□□□□□□         □□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□         □□□□□□□■■■■■□□□□□□□□□□□□□□□□
□□□□□□□□■■■■■■□□□□□□□□□□□□□□   →    □□□□□□□■□□□■□□□□□□□□□□□□□□□□
□□□□□□□□■□□□□■□□□□□□□□□□□□□□         □□□□□□□■■■■■□□□□□□□□□□□□□□□□
□□□□□□□□■■■■■■□□□□□□□□□□□□□□         □□□□□□□□□□□□□□□□□□□□□□□□□□□□
□□□□□□□□□□□□□□□□□□□□□□□□□□□□         □□□□□□□□□□□□□□□□□□□□□□□□□□□□
(The "8" digit, lots of wasted blank space)    (Focused on the digit only)
```

---

## 4. Feature Extraction

Feature extraction reduces the number of features by creating **new, compressed features** that still capture most of the original information.

**Key Difference:**
- **Feature Selection** → picks *existing* features and throws the rest away.
- **Feature Extraction** → creates entirely **new** compressed features from the old ones.

**Example — PCA (Principal Component Analysis):**

PCA is the most popular feature extraction technique. It takes a dataset with many dimensions (e.g., X, Y, Z, ...) and finds new axes (**Principal Components**) that capture the most spread (variance) in the data.

**Before PCA — Original 2D Data (X and Y dimensions):**

```
   Y
   |         ●  ●
   |      ●     ●
   |   ●     ●
   |      ●
   └────────────── X

Data is spread across both X and Y axes (2 dimensions needed)
```

**After PCA Transformation — Data compressed to 1D Principal Component:**

```
Step 1: Find the direction of maximum variance (the diagonal)

   Y                          ↗ PC1 (new axis that captures most spread)
   |     ●  ●              ↗
   |   ●    ●           ↗●●●●●●
   |  ●   ●          ↗
   | ●               
   └──────── X

Step 2: Project all points onto PC1

PC1 Dimension:
──●──●────●──●──●──●──►
  (All data now in 1 dimension instead of 2!)
```

**PCA on MNIST — Actual Real-World Example:**

The MNIST dataset has 784 pixel features (dimensions). After PCA:

| Before PCA | After PCA |
|---|---|
| 784 features / dimensions | ~50 Principal Components |
| Slow to train | Much faster to train |
| Hard to visualize | Can visualize in 2D/3D |
| Lots of noise | Noise is reduced |

Visualized in 2D using the top 2 Principal Components, similar digit images cluster together:

```
PCA 2nd Dimension (PC2)
   |
   |  [cluster of 0s]     [cluster of 1s]
   |       ●●●●                 ●●
   |     ●●●●●●●               ●●●
   |                [cluster of 8s]
   |                    ●●●●●
   └──────────────────────────── PCA 1st Dimension (PC1)
```

> The data is rotated and projected onto new axes (**Principal Components**) that capture the maximum variance. You can then drop the lower-variance components, effectively compressing 784 features into 50 while losing minimal information!

---

## Quick Revision Summary

| Technique | What it Solves | Key Method |
|---|---|---|
| **Missing Value Imputation** | `NaN` values that break models | Fill with mean/median/mode |
| **Handling Categorical Values** | Text that models can't read | Label Encoding, One-Hot Encoding |
| **Outlier Detection** | Extreme values that skew models | IQR, Z-score, Box Plot |
| **Feature Scaling** | Features on different scales | Normalization (Min-Max), Standardization |
| **Feature Construction** | Missing useful information | Create new columns from domain knowledge |
| **Feature Selection** | Too many irrelevant features | Keep only the most correlated features |
| **Feature Extraction** | Too many dimensions (Curse) | PCA — compress into Principal Components |
