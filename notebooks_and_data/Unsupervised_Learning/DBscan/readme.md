## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

This project includes a custom implementation of the DBSCAN algorithm.  
The main purpose of DBSCAN is to **find clusters in data** based on density rather than predefined shapes or counts.

### When to Use DBSCAN?
DBSCAN:
- Works well for **non-linearly separable clusters** (e.g., concentric circles, irregular shapes).
- Does not require specifying the number of clusters in advance.
- Can handle situations where k-means fails, such as when clusters vary in size, density, or are not spherical.
- Identifies **outliers/noise points** naturally, instead of forcing them into clusters.

## When Not to Use DBSCAN

While DBSCAN is powerful for discovering clusters of arbitrary shape, there are situations where it does not perform well:

- **Varying cluster densities**  
  DBSCAN assumes clusters are defined by a single density threshold (`eps`). If your dataset has clusters with very different densities, DBSCAN may merge them incorrectly or split them into noise.

- **High-dimensional data**  
  In many dimensions, distance metrics become less meaningful (the "curse of dimensionality"). DBSCAN struggles because neighborhood queries (`eps`) no longer reflect true similarity. Using PCA first can alleviate parts of this problem.

- **Clusters of very different sizes**  
  DBSCAN can fail when some clusters are small and dense while others are large and sparse, since a single `eps` cannot capture both.

- **Data with strong global gradients**  
  If points are distributed along a continuous gradient rather than forming dense pockets, DBSCAN may classify most points as noise or one giant cluster.


### How the Algorithm Works
1. **Neighborhood search**: For each point, DBSCAN finds all other points within a given radius (`eps`).
2. **Core points**: If a point has at least `min_samples` neighbors within `eps`, it is considered a core point.
3. **Cluster expansion**: Starting from a core point, DBSCAN recursively adds all reachable points within `eps` to the cluster.
4. **Noise handling**: Points that are not part of any cluster are labeled as noise.

### Key Characteristics
- **Density-based**: Clusters are formed where points are densely packed together.
- **Shape flexibility**: Can discover clusters of arbitrary shape, unlike k-means which assumes spherical clusters.
- **Robustness**: Resistant to outliers and noise in the dataset.

### Usage
This implementation can be applied to any numeric dataset.  

## Functionality
The core **dbscan** function performs density-based clustering on a provided dataset.  
It processes the input data to assign cluster labels to each data point, identifying noise where appropriate.

## Parameters
- **X (array_like)**:  
  The input data of shape `(n_samples, n_features)`.  
  Must be a 2-dimensional numeric array containing no NaN values.

- **eps (float)**:  
  The neighborhood radius. Defines the maximum distance between two samples for them to be considered neighbors.  
  Must be greater than 0.

- **min_samples (int)**:  
  The minimum number of points required in a neighborhood (including the point itself) to form a dense region (core point).  
  Must be greater than 0.

- **distance_func (callable, optional)**:  
  A function to calculate the distance between two vectors.  
  Must return a numeric scalar. Defaults to `EuclideanDistance`.

## Returns
- **labels (np.ndarray)**:  
  An array of shape `(n_samples,)` containing the computed cluster labels:
  - Non-negative integers: Indicate the cluster ID (e.g., 0, 1, 2).
  - `-1`: Indicates the point is noise (an outlier) and does not belong to any cluster.

## Error Handling
The function validates inputs and raises specific errors for invalid configurations:
- **ValueError**: Raised if `eps` or `min_samples` are less than or equal to 0.
- **TypeError**: Raised if `distance_func` is not callable or fails to return a numeric value.
