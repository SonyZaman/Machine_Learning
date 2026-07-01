# Bivariate & Multivariate Analysis

**Definition:** While Univariate analysis looks at one variable at a time, **Bivariate Analysis** examines the relationship between *two* variables (e.g., Does age affect survival?). **Multivariate Analysis** takes this even further by examining three or more variables simultaneously (e.g., Does age affect survival differently for men vs women?).

The code in `03_Bivariate_and_Multivariate_Analysis.ipynb` demonstrates how to choose the right visualization based on the **types** of data you are comparing (Numerical vs Categorical).

---

## 1. Scatterplot (Numerical - Numerical)
**What it does:** Plots individual data points on an X and Y axis. 
**Why use it:** It is the absolute best way to see if there is a correlation or mathematical relationship between two continuous numbers (e.g., as the bill goes up, does the tip go up?).
```python
import seaborn as sns
sns.scatterplot(data=tips, x='total_bill', y='tip')
```

## 2. Bar Plot (Numerical - Categorical)
**What it does:** Calculates and plots the aggregate (usually the average/mean) of a numerical column for each distinct category.
**Why use it:** To compare the average of a number across different groups (e.g., comparing the average fare paid by each passenger class).
```python
# Bivariate
sns.barplot(x=titanic['Pclass'], y=titanic['Age'])

# Multivariate (Adding 'hue' to split it by Sex as well)
sns.barplot(x=titanic['Pclass'], y=titanic['Fare'], hue=titanic['Sex'])
```

## 3. Box Plot (Numerical - Categorical)
**What it does:** Shows the 5-number statistical summary (min, 25%, median, 75%, max) and outliers for a numerical column, broken down by category.
**Why use it:** A bar plot only shows the average; a box plot shows the *entire distribution* and spread of the data for each group side-by-side.
```python
# Bivariate
sns.boxplot(x=titanic['Survived'], y=titanic['Age'])

# Multivariate
sns.boxplot(x=titanic['Survived'], y=titanic['Age'], hue=titanic['Sex'])
```

## 4. KDE Plot (Numerical - Categorical)
**What it does:** Draws smooth probability density curves for different categories on the same graph.
**Why use it:** Excellent for comparing the shapes of distributions. For example, comparing the age bell-curve of people who survived vs people who died to see where they overlap.
```python
# Using hue automatically draws overlapping curves for each category!
sns.kdeplot(data=titanic, x='Age', hue='Survived')
```

## 5. HeatMap (Categorical - Categorical)
**What it does:** Uses a color scale to represent numbers in a 2D matrix (often a crosstab).
**Why use it:** It is perfect for seeing the intersection of two categories (e.g., how many 1st class passengers survived vs died). Darker/lighter colors instantly highlight the highest concentrations.
```python
import pandas as pd
# First, create a matrix using crosstab, then plot the heatmap
matrix = pd.crosstab(titanic['Pclass'], titanic['Survived'])
sns.heatmap(matrix)
```

## 6. ClusterMap (Categorical - Categorical)
**What it does:** Similar to a heatmap, but it mathematically groups (clusters) similar rows and columns together using hierarchical clustering.
**Why use it:** If you have many categories, a clustermap automatically reorganizes the matrix so that categories that behave similarly are placed next to each other.
```python
sns.clustermap(pd.crosstab(titanic['Parch'], titanic['Survived']))
```

## 7. Pairplot (Automatic Multivariate)
**What it does:** Automatically plots a grid of scatterplots for *every single numerical column* against *every other numerical column* in your dataset.
**Why use it:** It is the ultimate "cheat code" for exploratory data analysis. With one line of code, you can instantly see all relationships in your dataset to figure out what is worth analyzing deeper.
```python
# Plots everything against everything!
sns.pairplot(iris, hue='species')
```

## 8. Lineplot (Numerical - Numerical / Time Series)
**What it does:** Connects data points with a continuous line.
**Why use it:** Primarily used for **Time Series** data (when the X-axis is time/years/months). It clearly shows trends, spikes, or drops over a specific period.
```python
# Grouping by year and summing the passengers before plotting
new = flights.groupby('year')['passengers'].sum().reset_index()
sns.lineplot(data=new, x='year', y='passengers')
```
