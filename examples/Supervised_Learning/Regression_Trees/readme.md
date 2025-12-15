# 📖 Overview
This repository contains a custom implementation of a Regression Tree, a supervised machine learning algorithm designed for predicting continuous numerical values. Unlike classification trees which predict discrete labels, this model recursively partitions the data to minimize error (Impurity) in the target variable.

It is designed with flexibility in mind, allowing users to optimize for different loss functions (Mean Squared Error vs. Mean Absolute Error) by injecting custom impurity metrics and leaf aggregators.

---

# 🧠 The Algorithm
The Regression Tree builds a predictive model by constructing a binary tree. It breaks down the dataset into smaller subsets while incrementally developing the decision logic.

## How It Works
- **Root Node:** The process begins with the entire dataset.
- **Splitting:** The algorithm iterates through every feature and every unique value to find the best split.
- **Criteria:** The "best" split is the one that maximizes Information Gain. In regression, this is calculated as the reduction in impurity:

  $$
  \text{Gain} = \text{Impurity(Parent)} - \sum \frac{N_{\text{child}}}{N_{\text{parent}}} \times \text{Impurity(Child)}
  $$

- **Recursive Partitioning:** The data is split into left and right child nodes, and the process repeats recursively.
- **Stopping:** Recursion stops when max_depth is reached, min_samples_split is not met, or impurity cannot be reduced further.
- **Prediction:** The final nodes (leaves) return a single scalar value derived from the samples in that node (e.g., the Mean or Median).

---

# 📉 Impurity Metrics & Objectives
A key feature of this implementation is the ability to choose the "Impurity" metric, which changes the mathematical objective of the tree.

## 1. Variance (Mean Squared Error)
**Use Case:** The default setting. Best for standard regression tasks where you want to minimize the Mean Squared Error (MSE).  
**The Logic:** Large errors are penalized heavily.  
**Leaf Aggregation:** Predictions are made using the Mean (np.mean) of samples in the leaf.  
**Formula:**  $$Var(y) = \frac{1}{n} \sum (y_i - \bar{y})^2$$

## 2. Mean Absolute Deviation (Mean Absolute Error)
**Use Case:** Best for data with outliers or when you want to minimize Mean Absolute Error (MAE).  
**The Logic:** Robust to extreme values; an outlier won't skew the prediction as drastically as it would with variance.  
**Leaf Aggregation:** Predictions are made using the Median (np.median) of samples in the leaf.  
**Formula:**  $$MAD(y) = \frac{1}{n} \sum |y_i - \text{median}(y)|$$

---

# 📊 When to Use It

## ✅ Strengths
- Non-Linear Relationships: Captures complex patterns without needing data transformation.
- No Feature Scaling: Works naturally with raw data of varying scales.
- Interpretable: The decision logic is transparent (white-box model).
- Flexible Objective: Can be tuned to be robust to outliers by switching to MAD/Median.

## ⚠️ Limitations
- Extrapolation: Cannot predict values outside the range of the training data.
- Instability: Small changes in data can lead to different tree structures.
- Computational Cost: The split-finding process iterates through all unique values, which can be slower on very large datasets compared to linear models.

---

# 🛠️ Usage & API

## `__init__(min_samples_split=2, max_depth=100, criterion=variance, leaf_aggregator=np.mean)`
Initializes the model. This is where you define the "strategy" of the tree.

### Parameters:
- **min_samples_split (int):** The minimum number of samples required to split an internal node. (Prevents overfitting).
- **max_depth (int):** The maximum depth of the tree.
- **criterion (Callable):** The function used to measure impurity.  
  - Pass variance for MSE optimization.  
  - Pass mean_absolute_deviation for MAE optimization.
- **leaf_aggregator (Callable):** The function to calculate the prediction at a leaf.  
  - Use np.mean if criterion is variance.  
  - Use np.median if criterion is mean absolute deviation.

---

## `fit(X, y)`
Trains the regression tree model on the provided dataset.

### Parameters:
- **X:** Numpy array of shape (n_samples, n_features). The training features.
- **y:** Numpy array of shape (n_samples,). The continuous target values.

### Returns:
- **self** (the fitted model).

---

## `predict(X)`
Generates predictions for a new set of data.

### Parameters:
- **X:** Numpy array of shape (n_samples, n_features).

### Returns:
- Numpy array of shape (n_samples,) containing the predicted values.
