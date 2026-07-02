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

ML models only understand **numbers**, not text like "Berlin", "Lisbon", or "Yes/No". We must convert these text categories into numerical representations.

**Example Dataset with Categorical Columns:**

| ID | City   | Age | Native? | Cat | Married? |
|----|--------|-----|---------|-----|----------|
| 1  | Lisbon | 25  | 0       | 0   | 0        |
| 2  | Berlin | 25  | 1       | 1   | 0        |
| 3  | Lisbon | 30  | 0       | 0   | 0        |
| 4  | Lisbon | 20  | 0       | 0   | 1        |
| 5  | Berlin | 10  | 0       | **6** | 0      |
| 6  | Berlin | NaN | 1       | 1   | 0        |
| 7  | Berlin | 16  | 0       | 1   | 0        |
| 8  | Berlin | 25  | 0       | 0   | 0        |
| 10 | Madrid | 25  | 1       | 0   | 1        |

> **Key Insight:** Column `Cat` has a value of `6` in row 5 — this is an **outlier** that needs to be detected and handled!

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

If `y` is Salary (e.g., 32,000 vs 33,000), its squared difference dominates the calculation completely, making the Age column irrelevant!

**The Solution — Normalize/Scale everything to the range `-1 to +1`:**

This ensures every feature contributes equally to the distance calculation.

---

## 2. Feature Construction

Feature construction means creating **brand new columns** from the existing data using domain knowledge. These new features often help the model find patterns it couldn't see before.

**Example — Titanic Dataset (Original):**

| ID | Country | Pclass | Name                                           | Sex    | Age | Nbsp | Fare   | Embarked |
|----|---------|--------|------------------------------------------------|--------|-----|------|--------|----------|
| 1  | 0       | 3      | Braund, Mr. Owen Harris                        | male   | 23  | ...  | A/5 3.2 | 6       |
| 2  | 1       | 1      | Cumings, Mrs. John Bradley (Florence Sing...)  | female | 22  | 1    | PC 17599 71.83 | C |

> **Example of Feature Construction:** From `Name`, you can extract the **Title** (Mr., Mrs., Miss., Dr.) as a new categorical feature. From `Age`, you can create `Age_Group` (Child, Teen, Adult, Senior). These new engineered features often have stronger predictive power than the raw originals!

---

## 3. Feature Selection

Feature selection is the process of choosing **only the most relevant features** and removing redundant or irrelevant ones.

**Why?** Using too many features can cause:
- **Overfitting** — the model memorizes the training data instead of learning patterns.
- **Slow training** — more features = more computation.
- **The Curse of Dimensionality** — in high dimensions, data becomes very sparse.

**Visual Example — MNIST Digit Recognition:**

The MNIST dataset contains images of handwritten digits (0-9). Each image is made of pixels, giving hundreds of features. Feature selection / extraction helps identify which pixels are actually important for recognizing a digit, and discards the empty/irrelevant background pixels.

```
Raw Image (28x28 = 784 features)  →  Select only important pixel features
        ∞                                         ↓
      [8]  (handwritten)               ~Reduced feature set~
```

---

## 4. Feature Extraction

Feature extraction reduces the number of features by creating **new, compressed features** that still capture most of the original information.

**Key Difference:**
- **Feature Selection** → picks existing features and throws the rest away.
- **Feature Extraction** → creates entirely **new** compressed features from the old ones.

**Example — PCA (Principal Component Analysis):**

PCA is the most popular feature extraction technique. It takes a dataset with many dimensions (e.g., X, Y, Z, ...) and compresses it into fewer **Principal Component** dimensions while retaining maximum variance.

```
Original High-Dimensional Space:
   Y
   |         ●  ●
   |      ●     ●
   |   ●     ●
   └────────────── X

         ↓ PCA Transformation

PCA 2nd Dimension (Y)
   |
   |     ●●●●●●●
   |
   └──────────────── PCA 1st Dimension (X)
```

> The data is rotated and projected onto a new axis (the **Principal Component**) that captures the most variance. You can then drop the 2nd dimension, effectively compressing 2D data into 1D while losing minimal information!

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
