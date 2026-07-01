# Step 1: Understanding Your Data

Before we can clean data or build machine learning models, we must intimately understand the dataset we are working with. The `01_Understanding_Data.ipynb` notebook demonstrates the essential first step of any Data Science project: **Data Profiling**.

Here is how and why we ask these specific questions when we first load a dataset:

## 1. What is the scale of our problem? (`df.shape`)
We first need to know how many rows (samples) and columns (features) we have. This tells us if we have enough data to train a complex model, or if we need more data. It also gives us an idea of the computational power required.

## 2. What does the raw data look like? (`df.sample(5)`)
While `df.head()` shows the first 5 rows, `df.sample()` pulls random rows. Looking at random samples helps us see the actual real-world values. We check: *Do the columns make sense? Are there obvious formatting issues? Does the data match our expectations?*

## 3. How is the computer interpreting our data? (`df.info()`)
Data comes in different types (integers, floats, text/objects). `df.info()` tells us if Pandas correctly identified numbers as numbers, or if a numerical column is accidentally being treated as text (which happens often when there are hidden characters or typos).

## 4. How dirty is our data? (`df.isnull().sum()`)
Real-world data is almost never perfect. By checking for missing values (`NaN`), we know exactly which columns will require imputation (filling in missing data) or dropping during the data cleaning phase.

## 5. What is the mathematical distribution? (`df.describe()`)
This generates a 5-number statistical summary for numerical columns. We look at the `min`, `max`, and `mean` to spot impossible outliers (e.g., an age of 250 years, or a negative fare price). 

## 6. Are there identical records? (`df.duplicated().sum()`)
Duplicate rows can artificially skew our model by giving too much weight to a specific data point. We check for duplicates to ensure data integrity before training.

## 7. Which features are actually important? (`df.corr()['Survived']`)
By checking the Pearson correlation coefficient against our target variable (e.g., `Survived`), we get an early mathematical hint about which features strongly influence the outcome. This guides our Feature Engineering later on!
