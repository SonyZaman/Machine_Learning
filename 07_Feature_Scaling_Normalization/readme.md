# Feature Scaling: Normalization

Normalization is a critical data preparation technique in machine learning. 

**The Goal:** Change the values of numeric columns in the dataset to use a common scale, *without distorting differences in the ranges of values or losing information*.

---

## 1. Types of Normalization

While Standardization (Z-score scaling) is one approach, Normalization offers several different techniques depending on the shape of your data:

1. **MinMax Scaling** (The most common, often just called "Normalization")
2. **Mean Normalization**
3. **Max Absolute Scaling**
4. **Robust Scaling**

*(This guide focuses on MinMax Scaling, as it is the most widely used form of Normalization).*

---

## 2. MinMax Scaling (Intuition & Math)

MinMax Scaling transforms all your numerical data to fall exactly within a specific range, usually **`[0, 1]`**.

### The Formula:
For every single data point ($x_i$), we calculate its new scaled value ($x_i'$) using this formula:

```
        X_i  -  X_min
 X_i' = ──────────────
        X_max - X_min
```
*Where:*
- **$X_i$** = The original data point
- **$X_{min}$** = The absolute minimum value in that column
- **$X_{max}$** = The absolute maximum value in that column

### Example Calculation (Weight):
Imagine we have a `Weight` column where the lightest person is **32 kg** and the heaviest is **130 kg**.
- $X_{min} = 32$
- $X_{max} = 130$

**Scaling the Maximum Value (130 kg):**
```
        130 - 32     98
 X_i' = ──────── = ──── = 1
        130 - 32     98
```
**Scaling the Minimum Value (32 kg):**
```
         32 - 32      0
 X_i' = ──────── = ──── = 0
        130 - 32     98
```
> **Result:** The heaviest person becomes exactly `1.0`, the lightest becomes exactly `0.0`, and everyone else falls somewhere in between!

---

## 3. Visualizing MinMax Scaling

Here is a visual representation of what happens to a scatter plot of Weight vs Height before and after MinMax Scaling:

```text
       BEFORE MIN-MAX SCALING                       AFTER MIN-MAX SCALING
       (Data is in original units)                  (Data is bounded within [0,1] box)
                                                            
   Height                                         Height' 
     │                                              │      
     │                                            1 ┼────────╭───────╮
     │        [Original Data Cluster]               │        │ x   x │ 
     │          x   x   x   x                       │        │ x x x │
     │        x   x   x   x   x                     │        │ x   x │ 
     │          x   x   x   x                       │        ╰───────╯
     │                                              │        │       │
     └─────────────────────────────── Weight        └────────┼───────┼─────── Weight'
    (0,0)                                         (0,0)      0       1
```

### Key Takeaways from the Visual:
1. **The Shape is Preserved:** Just like Standardization, the relative arrangement of the 'x' data points doesn't change. 
2. **The `[0, 1]` Bounding Box:** Unlike Standardization (which centers at 0 and spreads infinitely), MinMax Scaling literally builds a strict, hard boundary box. No data point will ever be less than 0 or greater than 1!
3. **Loss of Outlier Impact:** Because MinMax strictly uses the Max and Min values, outliers can heavily skew this box. (If one person weighs 500kg, all the normal people get squished into a tiny sliver between `0.0` and `0.1`).
