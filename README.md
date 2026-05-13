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
import plotly.graph_objects as go

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
