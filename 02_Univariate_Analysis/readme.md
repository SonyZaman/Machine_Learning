# Univariate Analysis

**Definition:** Univariate Analysis is the simplest form of analyzing data. "Uni" means "one", so in this step, we analyze data taking **one variable (column) at a time**. Our goal is not to find relationships between different variables (like age vs. survival), but rather to understand the distribution, frequency, and central tendency of a single variable in isolation.

The code in `02_Univariate_Analysis.ipynb` demonstrates how we approach Univariate Analysis depending on the type of data we are looking at:

---

## 1. Analyzing Categorical Data
Categorical data consists of discrete groups or labels (e.g., Male/Female, Survived/Died, City names). We analyze them using frequency counts.

### a. Countplot
**What it does:** Creates a bar chart where the height of each bar represents the absolute count (frequency) of each category.
**Why use it:** To quickly see which category is the most or least common.
```python
import seaborn as sns
sns.countplot(df['Embarked'])
```

### b. Pie Chart
**What it does:** Represents the categories as slices of a pie, showing their relative percentage of the whole.
**Why use it:** To easily visualize the proportion (e.g., "What percentage of passengers embarked from 'S'?").
```python
df['Embarked'].value_counts().plot(kind='pie', autopct='%.2f')
```

---

## 2. Analyzing Numerical Data
Numerical data consists of continuous measurable numbers (e.g., Age, Fare, Height). We analyze them by looking at their distribution and spread.

### a. Histogram
**What it does:** Groups continuous numbers into "bins" (ranges) and shows how many data points fall into each bin.
**Why use it:** To see the general shape of the data spread.
```python
import matplotlib.pyplot as plt
plt.hist(df['Age'])
```

### b. Histplot with KDE
**What it does:** Draws a histogram with a smooth KDE (Kernel Density Estimation) curve overlaid on top of it.
**Why use it:** The smooth KDE curve estimates the shape of the data, making it much easier to see if the data follows a normal distribution (bell curve) or if it is skewed left/right, without the visual noise of the jagged histogram bars.
```python
sns.histplot(df['Age'], kde=True)
```

### c. Boxplot
**What it does:** Visualizes the 5-number statistical summary: Minimum, First Quartile (25%), Median (50%), Third Quartile (75%), and Maximum.
**Why use it:** It is the absolute best tool for instantly identifying **Outliers** (data points that fall far outside the normal range, shown as single dots outside the "whiskers").
```python
sns.boxplot(df['Age'])
```

### d. Mathematical Metrics
Instead of graphs, we can also extract specific mathematical numbers to understand the numerical distribution mathematically:
```python
df['Age'].min()   # Finds the absolute lowest value
df['Age'].max()   # Finds the absolute highest value
df['Age'].mean()  # Calculates the average
df['Age'].skew()  # Measures asymmetry (0 = perfect bell curve)
```
