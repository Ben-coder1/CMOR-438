# Ordinary Least Squares (OLS) Linear Regression

Ordinary Least Squares (OLS) Linear Regression is one of the most fundamental and widely used algorithms in machine learning and statistics. It attempts to model the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data.  

---

## 📌 Conceptual Overview

At its core, Linear Regression assumes that the output can be calculated as a weighted sum of the inputs plus a bias term. The goal of the algorithm is to find the specific set of weights (coefficients) that minimizes the error between the model's predictions and the actual data points.

---

## ✅ When It Works Well

- **Linear Relationships:** The algorithm shines when the change in the target variable is proportional to the change in the features (e.g., Height vs. Weight or Square Footage vs. House Price).  
- **Interpretability:** Unlike "black box" models (like Neural Networks), OLS provides interpretable coefficients. If the coefficient for "Age" is -2.5, you know exactly that for every year older, the target decreases by 2.5 units.  
- **Simple Baselines:** It is often the first model you should run to establish a baseline for performance.

---

## ⚠️ When It Struggles

- **Non-Linear Patterns:** If the data follows a curve (e.g., bacterial growth or sine waves), a straight line will underfit significantly.  
- **Outliers:** OLS minimizes the squared error. A single extreme outlier has a quadratic impact on the cost function, often "pulling" the regression line far away from the main cluster of data to satisfy that one point.  
- **Multicollinearity:** If two features are highly correlated (e.g., "Temperature in Celsius" and "Temperature in Fahrenheit"), it becomes mathematically impossible to distinguish their individual effects, leading to unstable solutions.

---

## 🎨 Feature Engineering & Subjectivity

While the algorithm itself fits straight lines (or hyperplanes), you are not limited to linear data. You can "bend" the model to fit curves by transforming your data before training. By adding polynomial features, you can model complex relationships while keeping the equation linear in terms of the coefficients:

- **Linear:** $y = \theta_0 + \theta_1 x$  
- **Quadratic:** $y = \theta_0 + \theta_1 x + \theta_2 x^2$ (Parabola)  
- **Interaction:** $y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 (x_1 \cdot x_2)$  

### The Challenge of Subjectivity

This flexibility introduces a significant challenge: **Model Selection**.  
The algorithm cannot tell you which transformations to use:

- Do I square the height?  
- Do I take the log of the price?  
- Do I multiply age by income?  

Picking the correct features adds a layer of subjectivity to the modeling process. Choosing too few results in underfitting; choosing too many (e.g., adding $x^{10}$ to a small dataset) results in massive overfitting.

---

## 📐 Mathematical Foundation

To find the optimal line, we define a Cost Function called the **Residual Sum of Squares (RSS):**

$$
J(\theta) = \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
$$

To minimize this cost, we do not use iterative loops (like Gradient Descent). Instead, we solve for the point where the derivative is zero using linear algebra. This yields the **Normal Equation**:

$$
\hat{\theta} = (X^T W X)^{-1} X^T W y
$$

Where:  

- $\hat{\theta}$: The vector of optimal weights (coefficients).  
- $X$: The matrix of input features (augmented with 1s for the intercept).  
- $y$: The vector of target values.  
- $W$: A diagonal matrix of sample weights (Identity matrix if unweighted).

---

## 🛠 This Implementation

This package provides a robust, NumPy-based implementation of the OLS algorithm described above.

## Key Features

- **Closed-Form Solution:** Uses the Normal Equation for exact results (no learning rate or epochs required).  
- **Weighted Least Squares:** Supports the `sample_weight` parameter, allowing you to assign different importance to different data points (useful for handling outliers or heteroscedasticity).  
- **Automatic Intercept:** Handles the bias term automatically via `fit_intercept=True`.  

## Methods

### `__init__(fit_intercept=True)`  
Initializes the Linear Regression model.  

- `fit_intercept` (bool, default=True)  
  - If True, the model calculates a y-intercept (bias term).  
  - If False, the model is forced to go through the origin (0,0).  

### `fit(X, y, sample_weight=None)`  
Computes the optimal parameters $\theta$ using the Normal Equation.  

- `X`: 2D array of shape `(n_samples, n_features)`  
- `y`: 1D or 2D array of targets  
- `sample_weight` (Optional): 1D array of weights  

### `predict(X)`  
Generates predictions for new data using the learned model: $\hat{y} = X \theta$.  

### `mean_squared_error(X, y_true)`  
A convenience method that:  

- Generates predictions for `X` internally  
- Compares them against `y_true`  