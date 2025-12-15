## 📖 What kNN Does
The **k-Nearest Neighbors (kNN)** algorithm is a simple yet powerful method for classification and regression.  
When making a new prediction:
1. It calculates the distance between the new data point and all points in the training set.
2. It identifies the **k closest points** (neighbors) based on that distance metric.
3. It ignores the labels during distance calculation, but once neighbors are found, it looks at their labels.
4. The new point is assigned the label that represents the **majority vote** among those neighbors (for classification).  
   - For regression tasks, the prediction is the **average value** of the neighbors.

---

## ✅ Benefits of kNN
- **Intuitive and easy to implement**: No complex training phase; predictions are based directly on stored data.
- **Flexible decision boundaries**: Works well for data that is **nonlinearly separated**, since the decision boundary adapts to the local neighborhood.
- **Versatile**: Can be used for both classification and regression tasks.
- **No assumptions about data distribution**: Unlike parametric models, kNN does not assume linearity or specific distributions.

---

## ⚠️ Problems with kNN
- **High-dimensional data**: Distances become less meaningful in many dimensions (the "curse of dimensionality"), reducing accuracy.
- **Sparse data**: If data is spread out, neighbors may be far away, making predictions unreliable.
- **Computationally expensive**: Requires calculating distances to all training points at prediction time, which can be slow for large datasets.
- **Sensitive to irrelevant features**: Features that don’t matter can distort distance calculations unless proper scaling or feature selection is applied.
- **Choice of k matters**: Too small a k can lead to noisy predictions; too large a k can oversmooth and ignore local structure.

---


# Functionality
The **KNN** class provides a strict, numeric-only implementation of the kNN algorithm.  
Unlike generic implementations, this class enforces homogeneous numeric data for both feature vectors and labels, ensuring type safety and consistency.

---

## Class Interface
### `KNN(X=None, y=None)`
Initializes the model. You can store training data (`X`, `y`) during initialization or provide it later during prediction.

---

## Methods

### `predict(target, classify=True, K=5, X = None, y = None, dist=EuclideanDistance)`
**Purpose:** Specific prediction for a single target vector.  

**Parameters:**
- **target**: The input vector to predict.  
- **classify (bool)**: If `True`, performs classification (majority vote). If `False`, performs regression (average).  
- **K (int)**: Number of neighbors to consider.
- **X**, *y*: Training data used for prediction

**Returns:** The predicted label (scalar).

---

### `error(X_test, y_test, classify=True, K=5, ...)`
**Purpose:** Evaluates the model's performance against a test dataset.  

**Logic:**
- If `classify=True`: Returns the **Misclassification Rate** (0.0 to 1.0).  
- If `classify=False`: Returns the **Mean Absolute Error (MAE)**.  

---


## Data Constraints & Error Handling
This implementation includes strict validation logic:

- **Homogeneous Data:** Inputs must be numeric arrays. Ragged lists or mixed types (strings/numbers) are not supported.  
- **Validation:** Raises `ValueError` if array dimensions mismatch or if `K` exceeds the dataset size.  
- **Sanitization:** Automatically rejects input containing `NaN` values.
