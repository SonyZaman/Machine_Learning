# 01 - Understanding Data (Revision Notebook)

This README serves as a quick, well-organized revision guide for the initial data exploration steps covered in `01_Understanding_Data.ipynb`. Use this to quickly review the core Pandas functions anytime!

## Preparation: Loading the Data
Before asking any questions, we load our dataset (the Titanic dataset) using Pandas.
```python
import pandas as pd
df = pd.read_csv('train.csv')
```

---

## 1. How big is the data?
To find out the total number of rows and columns in the dataset:
```python
df.shape
```
*(Returns a tuple: `(rows, columns)`)*

## 2. How does the data look like?
To get a quick glimpse of random rows in the dataset, instead of just the top rows (`df.head()`), we use:
```python
df.sample(5)
```
*(Returns 5 random rows from the dataframe)*

## 3. What is the data type of cols?
To see the data types of all columns, the number of non-null values, and memory usage:
```python
df.info()
```

## 4. Are there any missing values?
To check for missing data and get the exact count of missing (`NaN`) values in every column:
```python
df.isnull().sum()
```

## 5. How does the data look mathematically?
To get a statistical summary (count, mean, standard deviation, min, max, and percentiles) for all numerical columns:
```python
df.describe()
```

## 6. Are there duplicate values?
To find out exactly how many identical duplicate rows exist in the dataset:
```python
df.duplicated().sum()
```

## 7. How is the correlation between cols?
To find the mathematical relationship (Pearson correlation) between all numerical columns and our target column (`Survived`):
```python
df.select_dtypes(include=['number']).corr()['Survived']
```
*(A positive value means they increase together, a negative value means as one goes up the other goes down)*

---
*Happy Machine Learning!*
