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
