# 📈 Logistic Regression: A Classifier for Binary Outcomes

Logistic Regression is a foundational yet powerful statistical model used for binary classification problems. Despite its name, it is a classification algorithm, not a regression algorithm. It models the probability of a given input belonging to a specific class.

---

## Purpose

The fundamental purpose of Logistic Regression is to estimate the probability that an input sample belongs to the positive class (labeled **1**) versus the negative class (labeled **0**).  
This probabilistic output is then converted into a discrete class prediction by applying a threshold (typically **0.5**).

**Goal:**  
To find a decision boundary (a hyperplane in multi-dimensional space) that best separates the two classes in the feature space.

---

## How it Works: The Mechanism

Logistic Regression combines a linear model with a non-linear function called the **Sigmoid function** (or Logistic function).

### **1. The Linear Model (The Line)**

First, the model calculates a linear combination of the input features ($X$) and the learned coefficients (weights, $\mathbf{w}$), plus an intercept ($b$):

$$
z = \mathbf{w}^T \mathbf{X} + b
$$

### **2. The Sigmoid Function (The Squish)**

The value $z$ can range from $-\infty$ to $+\infty$. To map this output to a probability between 0 and 1, the Sigmoid function ($\sigma$) is applied:

$$
\hat{p} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

This $\hat{p}$ (the predicted probability) is the model's confidence that the input $X$ belongs to class 1.

### **3. Training via Gradient Descent**

The model is trained by minimizing the **Binary Cross-Entropy (Log Loss)** function, which penalizes the model when its predicted probability $\hat{p}$ is far from the true label $y$.  
This minimization is done iteratively using **Gradient Descent**, which adjusts the weights $\mathbf{w}$ and intercept $b$ in the direction that most reduces the loss.

---

## Strengths and Weaknesses

### **What it's Good For**

- Probability Output: Provides well-calibrated probabilities for risk assessment.  
- Interpretability: The coefficients $\mathbf{w}$ are easy to interpret (e.g., $e^{\mathbf{w}_i}$ is the odds ratio).  
- Speed and Simplicity: Fast to train and computationally inexpensive.  
- Baseline Model: Serves as an excellent first model or baseline for comparison against more complex algorithms.

### **What it's Bad For**

- Linearity Constraint: Assumes a linear relationship between features and the log-odds of the outcome.  
- Feature Engineering Needed: Struggles with complex, non-linear boundaries.  
- Susceptible to Outliers: Can be strongly influenced by extreme data points.  
- High-Dimensional Data: Performance can degrade with sparse, high-dimensional data (though less so than some other models).

---

## Achieving Non-Linear Relationships

While Logistic Regression is intrinsically a linear model, its power can be extended to model non-linear boundaries through **feature engineering**.

The model only assumes that the relationship between the features and the **log-odds** is linear. If the true relationship between the features ($X$) and the outcome is quadratic or exponential, you can transform the input features before feeding them to the model.

For example, to capture a **parabolic decision boundary** (a non-linear relationship):

$$
z = \mathbf{w}_1 x_1 + \mathbf{w}_2 x_2 + \mathbf{w}_3 x_1^2 + b
$$

By transforming $x_1$ to $x_1^2$ and including it as a new feature, the model remains **linear in its parameters** ($\mathbf{w}$), but becomes **non-linear in its features**, allowing it to learn complex, curved boundaries.

---

# Implementation Functionality: LogisticRegression Class

This particular implementation uses **Gradient Descent** to train the model and is structured to handle standard binary classification tasks.

---

## `__init__` (Initialization)

The constructor sets the key hyperparameters and initializes the model's state:

- **learning_rate (float):** Controls the step size during optimization. A crucial parameter for convergence speed and stability.  
- **n_iterations (int):** The total number of times the Gradient Descent loop will run to refine the coefficients.  
- **fit_intercept (bool):** Determines whether the bias term ($b$) should be calculated and included in the model.

It initializes `self.coef_` (weights) and `self.intercept_` (bias) to `None`.

---

## `fit(X, y)` (Training)

This method trains the model using the provided data:

- **Validation:** Ensures input arrays are numeric, non-empty, and free of NaNs. It specifically validates that the target array `y` contains only binary values (0 or 1).  
- **Initialization:** Sets `self.coef_` to a zero vector and initializes `self.intercept_` (if `fit_intercept` is True).

### Gradient Descent Loop

Runs for `n_iterations`:

- Calculates the linear output $z$.  
- Applies the sigmoid function to get the predicted probabilities $\hat{p}$.  
- Calculates the error $(\hat{p} - y)$.  
- Calculates the partial derivatives (gradients) $d_{\mathbf{w}}$ and $d_b$.  
- Updates the coefficients and intercept by subtracting `learning_rate` times the gradient.

---

## `predict_proba(X)` (Probability Output)

Calculates and returns the raw probability of the positive class ($P(y=1 \mid X)$) for new data $X$:

- Uses the learned `self.coef_` and `self.intercept_` to compute $z$.  
- Applies the sigmoid activation to $z$ to return an array of probabilities in $[0, 1]$.

---

## `predict(X, threshold=0.5)` (Class Prediction)

This method converts the probabilities into final class labels:

- Calls `self.predict_proba(X)` to get probabilities.  
- Applies the specified threshold (default is 0.5):  
  - Probability ≥ threshold → class **1**  
  - Probability < threshold → class **0**