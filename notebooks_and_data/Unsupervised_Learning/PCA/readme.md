# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is a fundamental dimensionality reduction technique used to simplify complex datasets while retaining the most important information. It transforms a large set of correlated variables into a smaller set of uncorrelated variables called "principal components. Note that I place this algorithm in the pre_processing directory of the package.

## 📌 The Algorithm & Purpose

The primary purpose of PCA is feature extraction and noise reduction. By projecting data onto a lower-dimensional subspace, it reveals the hidden linear structures within the data. It is widely used for:

- **Preprocessing:** Preparing data for machine learning algorithms (like K-Means, DBSCAN, or Neural Networks) by removing redundant features and reducing computational cost.
- **Visualization:** Compressing high-dimensional data (e.g., 50 features) down to 2D or 3D so it can be plotted and inspected by humans.
- **Denoising:** Discarding the "tail" components that often represent random noise rather than signal.

## ⚠️ Where It Struggles

While PCA is a standard workhorse in data science, it has distinct limitations:

- **Linear Assumption:** PCA only finds linear correlations. It fails to capture complex, non-linear manifolds (e.g., a "Swiss Roll" dataset where points spiral).
- **Outliers:** Because it maximizes variance, PCA is highly sensitive to extreme outliers, which can skew the principal components.
- **Interpretability:** The new features are linear combinations of all original features. This "blurs" the meaning—you no longer have "Age" or "Income," but rather "Component 1" (a mix of both).

## ⚙️ How It Works (Conceptual)

The algorithm is purely mathematical and requires no iterative training. It follows this linear algebra pipeline:

1. **Centering:** The mean of each feature is subtracted from the dataset so that the cloud of data points is centered around the origin (0,0).
2. **Covariance Matrix:** The algorithm calculates how every variable relates to every other variable. This matrix captures the spread and correlation of the data.
3. **Eigendecomposition:**  
   - We compute the eigenvectors (directions) and eigenvalues (magnitude of variance) of the covariance matrix.  
   - **Eigenvectors** point in the directions where the data is most spread out.  
   - **Eigenvalues** tell us how much information (variance) that direction holds.
4. **Sorting:** The components are ranked by their eigenvalues. The top component captures the most variance, the second captures the second most, and so on.
5. **Projection:** The original data is multiplied by the top \( k \) eigenvectors to transform it into the new, lower-dimensional space.

## 🛠 This Implementation

This Python implementation provides a streamlined, NumPy-based wrapper for PCA, specifically designed for your preprocessing module. It focuses on safety and ease of integration with other ML tools.

### Functionality & Features

- **Preprocessing Focus:**  
  This tool is placed in the `preprocessing/` folder because it is best used as a preparatory step before feeding data into estimators like K-Means, KNN, or Perceptrons.

- **Robust Validation:**  
  - Includes internal checks (`_ensure_numeric_array`, `_ensure_no_nan`) to prevent silent failures with bad data types or missing values.  
  - Validates that `n_components` does not exceed the number of available features.

- **Flexible Projection:**  
  Allows you to choose any number of directions (`n_components`).  
  Returns the Explained Variance explicitly, helping you judge how much information was lost during reduction.

## API Reference

### Python

```python
components, explained_variance, transformed_data = compute_pca(data, n_components=None)
```

**data:** The 2D input array (samples × features).

**n_components:** The number of dimensions to keep. If `None`, all components are kept (useful for simple decorrelation).

## Returns

**components:** The direction vectors (eigenvectors).

**explained_variance:** The magnitude of variance for each component.

**transformed_data:** The dataset projected into the new space.