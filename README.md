# 📈 Salary Prediction using Ordinary Least Squares (OLS) Linear Regression

Welcome to my Machine Learning project where we explore the **Ordinary Least Squares (OLS)** algorithm! This project walks through the mathematical intuition behind Linear Regression by manually calculating the line of best fit, and then transitions into using standard industry libraries (`scikit-learn`) for both Simple and Multiple Linear Regression.

## 1) What is OLS? 🧮
**Ordinary Least Squares (OLS)** is the underlying mathematical method used in Linear Regression. 

Imagine you have a scatter plot of data points (like Work Experience vs. Salary). You want to draw a straight, continuous line through these points that best captures the trend. OLS calculates the best possible line by **minimizing the sum of the squares of the differences (residuals)** between the actual data points and the predicted points on the line. 

In simple terms, for the equation of a line `y = mx + c`:
* OLS finds the perfect **slope (m)** and **intercept (c)** so that the line makes the fewest errors in prediction.

## 2) What are we building? 🛠️
In this project, we are building a progressive Salary Prediction Model in three phases:
1. **The Manual Approach:** We manually calculate a "multiplication factor" (slope) and try guessing an intercept to predict salary based purely on work experience.
2. **Simple Linear Regression (1 Feature):** We use Python's `scikit-learn` library to automatically apply OLS and find the mathematically perfect slope and intercept for Work Experience vs. Salary.
3. **Multiple Linear Regression (Multiple Features):** We upgrade our model to predict a person's salary based on multiple real-world factors: **Experience, Age, and Performance Rating**.

## 3) Key Takeaways for Beginners 💡
* **ML is just Math under the hood:** Machine Learning isn't magic. As demonstrated in Phase 1 and 2, training a model just means calculating the perfect mathematical weights (slope and intercept).
* **Why use Scikit-Learn?** Manually guessing the parameters (like `p = 12500` and `c = 16000`) is tedious and prone to error. Scikit-learn's `model.fit()` runs the OLS math instantly to give us the absolute best parameters.
* **Scaling is easy:** The true power of ML libraries is how easily you can switch from 1 independent variable (Simple Regression) to multiple independent variables (Multiple Regression) using the exact same code structure.

---

## 4) Step-by-Step Code Walkthrough 🚀

### Step 1: The Manual Approach
First, we load our dataset, sort it, and manually find the average ratio of Salary to Work Experience.

```python
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample Data
work_exp = [8, 2, 18, 1, 10, 5, 14, 3, 6, 2, 20, 9, 4, 16, 5, 1, 12, 7, 3, 10,
            6, 17, 4, 8, 13, 7, 15, 9, 11, 19]
salary = [112500, 42500, 237500, 26667, 137500, 69167, 187500, 53333, 87500, 37500,
          266667, 125000, 58333, 212500, 75833, 31667, 162500, 100000, 48333, 131667,
          81667, 225000, 65000, 112500, 175000, 93333, 200000, 119167, 150000, 250000]

df = pd.DataFrame({'work_exp': work_exp, 'salary': salary})

# Finding the multiplication factor manually
m = (df['salary'] / df['work_exp']).mean()
print('Manual Multiplier (m):', m)

# Predicting for 50 years of experience
x = 50
print('Predicted Salary:', x * m)
```
**Expected Output:**
```text
Predicted Salay: 772681.2931217102
```

### Step 2: Visualizing Actual vs. Predicted Salary (Manual Approach)

Now we will plot our actual data points against the predictions we made using our manual multiplier (`m`). This helps us visually check how well our manual model is performing. 

```python

# 1. Calculate the predicted salary using our manual multiplier (m)
df['predicted_salary'] = df['work_exp'] * m

# 2. Create an empty Plotly figure
fig = go.Figure()

# 3. Add the Actual Salary points (Scatter Plot / Dots)
fig.add_trace(go.Scatter(
    x=df['work_exp'],
    y=df['salary'],
    mode='markers',
    name='Actual Salary'
))

# 4. Add the Predicted Salary points (Line Plot)
fig.add_trace(go.Scatter(
    x=df['work_exp'],
    y=df['predicted_salary'],
    mode='lines+markers',
    name='Predicted Salary'
))

# 5. Add titles and format the layout
fig.update_layout(
    title='Actual vs Predicted Salary (Manual Tuning)',
    xaxis_title='Work Experience (Years)',
    yaxis_title='Salary (₹)'
)

# 6. Display the graph
fig.show()
```
![Actual vs Predicted Salary (Manual)](newplot.png)

**🔍 Analyzing the Output:**

If you look closely at the graph above, you will notice a clear divergence between the **Actual Salary (blue dots)** and our **Predicted Salary (orange line)**. 

* **The Good:** For employees with 1 to 5 years of experience, our manual multiplier does a fairly decent job of estimating the salary.
* **The Bad:** As work experience increases, the orange line starts to heavily over-predict. By the 20-year mark, our manual model predicts a salary over ₹300,000, while the actual data point is closer to ₹260,000. 

**Why did this happen?** Because we only calculated a simple average ratio (our slope, `m`) without accounting for a baseline starting salary (the intercept, `c`). This creates a rigid line that multiplies our errors as the numbers get bigger. 

This growing gap (known mathematically as the *residual error*) perfectly illustrates why manual calculations aren't efficient for real-world data. In the next step, we will use the **Scikit-Learn** library to automatically calculate the exact slope and intercept that reduces these errors to the absolute minimum!


### Step 3: Improving the Model Manually (Adding an Intercept)

In the previous step, we only used a slope (our multiplier, `m`). This forces our prediction line to point straight towards zero, which isn't how real-world salaries work (even with 0 years of experience, a starting salary usually exists). 

To fix this, we need to complete the equation of a straight line: `Salary = (Work Experience * p) + c`.
* `p` = Our new manual slope 
* `c` = Our manual intercept (base starting value)

Let's test this by manually guessing `p = 12500` and `c = 16000` to see if our prediction line fits the data better!

```python

# 1. Define our manual slope (p) and intercept (c)
p = 12500
c = 16000

# 2. Calculate the predicted salary using the full line equation: y = px + c
df['predicted_salary'] = (df['work_exp'] * p) + c

# 3. Create the Plotly figure
fig = go.Figure()

# 4. Add the Actual Salary points
fig.add_trace(go.Scatter(
    x=df['work_exp'],
    y=df['salary'],
    mode='markers',
    name='Actual Salary'
))

# 5. Add the Predicted Salary points
fig.add_trace(go.Scatter(
    x=df['work_exp'],
    y=df['predicted_salary'],
    mode='lines+markers',
    name='Predicted Salary'
))

# 6. Add titles and format the layout
fig.update_layout(
    title='Actual vs Predicted Salary (Manual Slope & Intercept)',
    xaxis_title='Work Experience (Years)',
    yaxis_title='Salary (₹)'
)

# 7. Display the graph
fig.show()
```
![Actual vs Predicted Salary (Manual)](SECOND_GRAPH.png)

**🔍 Analyzing the Output:**

What a difference an intercept makes! If you compare this graph to the one in Step 2, the improvement is massive:

* **A Perfect Fit:** By adding the intercept (`c = 16000`), our prediction line is no longer forced to point towards zero. It now has a realistic baseline starting salary.
* **Errors Minimized:** The orange prediction line now beautifully tracks the actual blue data points across the *entire* range. That huge over-prediction gap we saw at the 20-year mark in Step 2 has completely disappeared!

This perfectly illustrates the power of the complete algebraic equation for a line: `y = mx + c`. 

**The Catch:** While our line looks great, we still just "guessed" the numbers `p = 12500` and `c = 16000`. Guessing works for simple datasets, but it is impossible for massive datasets with millions of rows. 

In **Step 4**, we will stop guessing and use the **Scikit-Learn** Machine Learning library to mathematically calculate the absolute perfect OLS parameters in a fraction of a second!
