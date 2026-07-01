# Understanding Data: Jupyter Notebook Markdown Cheat Sheet

This repository contains my machine learning journey. Below is a comprehensive revision notebook for all the Markdown syntax you can use inside Jupyter Notebook (`.ipynb`) Markdown cells!

---

## 1. Headings
Use `#` for headings. The number of `#` corresponds to the heading level (1-6).
```markdown
# Heading 1 (Largest)
## Heading 2
### Heading 3
#### Heading 4
##### Heading 5
###### Heading 6 (Smallest)
```

## 2. Text Emphasis
Use asterisks `*` or underscores `_` to style your text.
```markdown
*Italic Text* or _Italic Text_
**Bold Text** or __Bold Text__
***Bold and Italic*** or ___Bold and Italic___
~~Strikethrough Text~~
```

## 3. Lists
### Unordered Lists
Use `-`, `*`, or `+` for bullet points.
```markdown
- Item 1
- Item 2
  - Sub-item A (indent with 2 or 4 spaces)
  - Sub-item B
```

### Ordered Lists
Use numbers followed by a period.
```markdown
1. First step
2. Second step
   1. Sub-step A
   2. Sub-step B
```

### Task Lists (Checkboxes)
```markdown
- [x] Completed task
- [ ] Incomplete task
```

## 4. Links & Images
### Links
```markdown
[Click here for Google!](https://www.google.com)
```
### Images
Add an exclamation mark `!` before the link syntax.
```markdown
![Alt text for image](https://url-to-image.jpg)
```

## 5. Code & Syntax Highlighting
### Inline Code
Use a single backtick `` ` `` around the text.
```markdown
Here is an example of an `inline code` snippet.
```

### Code Blocks
Use triple backticks ` ``` ` and specify the language for syntax highlighting.
```markdown
\```python
import pandas as pd
df = pd.read_csv("train.csv")
print(df.head())
\```
```
*(Note: Remove the backslashes `\` in actual use, they are just for escaping here).*

## 6. Blockquotes
Use `>` to indent text as a quote.
```markdown
> This is a blockquote.
>> This is a nested blockquote.
```

## 7. Horizontal Rules (Dividers)
Use three or more hyphens, asterisks, or underscores to create a horizontal line.
```markdown
---
***
___
```

## 8. Tables
Use pipes `|` to separate columns and hyphens `-` for the header row.
```markdown
| Column 1 | Column 2 | Column 3 |
| :--- | :---: | ---: |
| Left-aligned | Center-aligned | Right-aligned |
| Row 2 | Row 2 | Row 2 |
```

## 9. Mathematical Equations (LaTeX)
Jupyter Markdown cells natively support MathJax (LaTeX) for mathematical formulas.
### Inline Math
Use single `$` signs.
```markdown
The equation of a line is $y = mx + c$.
```
### Block Math (Centered)
Use double `$$` signs.
```markdown
$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2
$$
```

## 10. HTML Formatting (For extra styling)
Since Jupyter cells render HTML, you can use HTML tags for custom colors and styling!
```markdown
<font color="red">This text is red!</font>
<div align="center">This text is centered!</div>
<b>This is bold HTML text</b>
<br> (This adds a line break)
```

---
*Keep this file open as a quick reference while writing beautiful documentation in your Jupyter Notebooks!*
