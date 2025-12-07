# Decision Tree

📋 **Overview**  
Decision Trees are a non-parametric supervised learning method used for classification. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

---

## **Key Capabilities**

**Disjoint / Fragmented Grouping:**  
This is a major advantage over linear models. The algorithm can successfully classify data where a single class is split into separate, disconnected regions.  
**Example:** If Class A exists in both the top-left and bottom-right corners, while Class B sits in the middle, the tree can isolate the middle region and correctly group the separated corners as the same class.

**Non-Linear Separation:**  
It does not assume a straight-line relationship between features. It solves complex problems by creating a "staircase" of decision boundaries.

**Multi-Criterion Support:**  
Supports both Entropy (Information Gain) and Gini Impurity for determining optimal splits.

---

## 🧠 **How It Works**

The algorithm builds a tree structure where:

- **Nodes** represent a decision based on a specific feature and threshold (e.g., `Weight <= 180`).  
- **Edges (branches)** represent the outcome of that decision (True/Left or False/Right).  
- **Leaves** represent the final class label prediction.

### **The Algorithm Steps**

1. **Start at the Root:** The algorithm takes the entire dataset.  
2. **Calculate Impurity:** It measures the "impurity" of the current set using Entropy or Gini.  
3. **Find Best Split:** It iterates through every unique value of every feature to find the split that maximizes Information Gain (decreases impurity the most).  
4. **Partition:** The data is split into left and right subsets based on the best threshold.  
5. **Recursion:** Steps 2–4 are repeated recursively for each subset until a stopping criterion is met.

---

## 🚀 **When to Use (and When Not To)**

### ✅ **When to Use**

- **Complex/Disjoint Distributions:** When your data has clusters of the same class separated by other classes (e.g., the "XOR" problem).  
- **Interpretability:** When you need to explain *why* a prediction was made to a human stakeholder.  
- **Mixed Data Scales:** Decision trees are generally invariant to scaling; you don't necessarily need to normalize your data (though this implementation enforces numeric arrays).

### ❌ **When Not to Use (Limitations)**

- **High-Dimensional Data:** Decision trees struggle with the "Curse of Dimensionality."  
- **Extrapolation:** They are poor at predicting outside the range of the training data.  
- **Small Variations:** Small changes in the data can result in a completely different tree structure (high variance).

---

## 📦 **API Documentation**

### **DecisionTree Class**  
The main entry point for training and prediction.

```python
class DecisionTree(min_samples_split=2, max_depth=100, criterion='entropy')
```

### Parameters

| Parameter          | Type            | Default     | Description |
|-------------------|-----------------|-------------|-------------|
| `min_samples_split` | int             | 2           | The minimum number of samples required to split an internal node. |
| `max_depth`         | int             | 100         | The maximum depth of the tree to prevent overfitting. |
| `criterion`         | str or Callable | 'entropy'   | The function to measure split quality. Options: 'entropy', 'gini'. |

---

### Core Methods

**`fit(X, y)`**  
Trains the model.  
- `X`: Training vectors of shape `(n_samples, n_features)`.  
- `y`: Target values of shape `(n_samples,)`.  

*Note:* Validates input shapes and ensures no NaNs exist.

**`predict(X)`**  
Predicts class labels for samples in `X`.  
Returns an array of predicted class labels.

---

### Node Class

Represents a single structural unit in the graph.

**Attributes:**
- **feature:** Index of the feature used for splitting.  
- **threshold:** Value to compare against.  
- **left / right:** Pointers to child nodes.  
- **value:** The predicted class (if the node is a Leaf).

---

### Impurity Metrics

This implementation includes two metric functions:

#### **entropy(y)**  
Calculates Shannon Entropy. High entropy means the dataset is mixed (impure); low entropy means it is mostly one class.  
Formula:

\[
H(S) = -\sum p_i \log_2(p_i)
\]

#### **gini(y)**  
Calculates Gini Impurity. A faster alternative to entropy that minimizes the probability of misclassification.  
Formula:

\[
G(S) = 1 - \sum p_i^2
\]