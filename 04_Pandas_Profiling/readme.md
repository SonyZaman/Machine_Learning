# Pandas Profiling (YData Profiling)

---

## What is Pandas Profiling?

**Pandas Profiling** (now officially renamed to **YData Profiling**) is an open-source Python module with which we can quickly do an Exploratory Data Analysis (EDA) with just a few lines of code.

Instead of manually writing dozens of lines of code to check for missing values, calculate correlations, and plot distributions, this library generates a comprehensive, interactive HTML report from a pandas DataFrame automatically.

---

## Key Features Generated in the Report

When you run a profile report, it automatically calculates and visualizes:

1. **Type Inference**: Detects the types of columns (Categorical, Numeric, Date, etc.)
2. **Essentials**: Type, unique values, missing values
3. **Quantile Statistics**: Minimum, Q1, median, Q3, maximum, range, interquartile range
4. **Descriptive Statistics**: Mean, mode, standard deviation, sum, median absolute deviation, coefficient of variation, kurtosis, skewness
5. **Most Frequent Values**
6. **Histograms**: Categorical and Numerical
7. **Correlations**: Highlighting of highly correlated variables, Spearman, Pearson, and Kendall matrices
8. **Missing Values Matrix, Count, Heatmap, and Dendrogram**

---

## Important Update (Pandas Profiling -> YData Profiling)

*Note: The original `pandas-profiling` package is deprecated and often causes dependency errors (like `PydanticImportError`) with modern Python environments. It has been completely replaced by `ydata-profiling`.*

**Installation:**
```bash
pip install ydata-profiling
```

---

## Code Implementation

Here is the exact code required to generate a comprehensive EDA report for the Titanic dataset (`train.csv`):

```python
import pandas as pd
from ydata_profiling import ProfileReport

# 1. Load the dataset
df = pd.read_csv('train.csv')

# 2. Generate the report
prof = ProfileReport(df, title="Titanic Dataset Profiling Report")

# 3. Export the report to an interactive HTML file
prof.to_file(output_file='output.html')
```

Once the code finishes running, an `output.html` file will be generated in your directory. 
You can open this file in any web browser (Chrome, Safari, Edge) to explore the full interactive EDA dashboard!
