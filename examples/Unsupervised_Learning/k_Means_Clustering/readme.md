# K-Means Clustering

K-Means is one of the most popular **unsupervised learning algorithms** used for partitioning a dataset into \(k\) distinct, non-overlapping subgroups (clusters).

---

## 📖 How It Works

The algorithm iteratively partitions the dataset into \(k\) pre-defined distinct non-overlapping subgroups (clusters), where each data point belongs to only one group.

1. **Initialization**  
   Randomly selects \(k\) data points as initial *centroids* (centers of the clusters).

2. **Assignment**  
   Each data point is assigned to the closest centroid based on a distance metric (usually Euclidean distance).

3. **Update**  
   The new centroid of each cluster is calculated by taking the mean of all data points assigned to that cluster.

4. **Repeat**  
   Steps 2 and 3 are repeated until the centroids stop moving (*convergence*) or a maximum number of iterations is reached.

---

## ✅ When to Use K-Means

- **General-purpose clustering**: Often the first algorithm to try for unsupervised learning tasks.  
- **Flat geometry**: Works best when clusters are spherical and roughly the same size and density.  
- **Large datasets**: Relatively efficient (\(O(n)\)) compared to hierarchical clustering (\(O(n^2)\)).  

---

## ⚠️ Limitations & Constraints

- **Choosing \(k\)**: Must specify the number of clusters in advance. Techniques like the *Elbow Method* can help.  
- **Sensitivity to Initialization**: Random starting points can lead to different results. (Using a seed helps reproducibility).  
- **Spherical Assumption**: Struggles with non-convex shapes (e.g., crescents, rings) or clusters of varying densities.  
- **Outliers**: Centroids can be dragged by outliers, or outliers might form their own cluster if \(k\) is large enough.  

---

## ⚙️ Functionality

The module provides the **`kmeans_clustering`** function, a robust implementation designed for numeric datasets:

```python
kmeans_clustering(X, k, epsilon=None, max_iter=None, seed=None, distance_func=None)
```
## ⚙️ Parameters

- **X (array_like)**: Input data of shape `(n_samples, n_features)`. Must be a 2D numeric array with no NaN values.  
- **k (int)**: Desired number of clusters. Must be between 2 and the total number of samples.  
- **epsilon (float, optional)**: Convergence threshold. Stops if centroid movement < epsilon.  
- **max_iter (int, optional)**: Hard limit on iterations. Stops after this many steps even if not converged.  
  > **Note:** You must provide at least one stopping condition (`epsilon` or `max_iter`).  
- **seed (int, optional)**: Random seed for reproducibility. Controls initial centroid selection.  
- **distance_func (callable, optional)**: Custom function to calculate distance between points.  
  - **Default:** Euclidean Distance.  
  - ⚠️ **Be cautious:** Non-Euclidean distances may cause suboptimal convergence.  

---

## 📤 Returns

- **centroids (np.ndarray)**: Array of shape `(k, n_features)` representing final cluster centers.  
- **labels (np.ndarray)**: Array of shape `(n_samples,)` containing the cluster index (0 to k-1) for each point.  

---

## 🔄 Distinct Behaviors

- **Empty Cluster Handling**: If a cluster ends up with zero members, the centroid is re-initialized to a random point from the dataset.  
- **Strict Validation**:  
  - Ensures \(2 \leq k \leq n_{samples}\).  
  - Distances must return numeric values.  
  - Dataset must contain no missing values (NaN).  

---
