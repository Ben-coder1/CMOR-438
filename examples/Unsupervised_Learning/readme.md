# Unsupervised Learning Examples

This folder contains example notebooks demonstrating Unsupervised Learning algorithms from our custom ML package.

---

## What is Unsupervised Learning?

Unsupervised learning is a critical branch of Machine Learning where the algorithms are trained using data that does **not** have labeled outcomes (i.e., no target variables or known classes).  
This makes the techniques particularly useful when you don't have labeled data or when the goal is to explore the inherent structure within the data itself.

Common tasks in unsupervised learning include:

- **Finding Clusters or Communities:** Grouping similar data points together.  
- **Dimensionality Reduction:** Reducing the number of features while preserving essential information.

---

## Algorithms Demonstrated

The notebooks in this directory provide practical demonstrations for the following unsupervised techniques:

### K-Means Clustering
A popular partition-based clustering algorithm that aims to partition \( n \) observations into \( k \) clusters, where each observation belongs to the cluster with the nearest mean.  
This algorithm primarily operates on numerical data.

### DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
A density-based clustering algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.  
This algorithm primarily operates on numerical data.

### Majority Label Propagation
A technique that uses the inherent structure (such as connectivity or similarity) of the data to spread information across the dataset.  
It can be used for both unsupervised tasks (finding implicit community structure) and semi-supervised tasks (using a few labeled points to infer labels for the rest of the data).  
Crucially, this algorithm operates on **graph data**, where data points are represented as nodes and relationships are represented as edges, unlike the others which operate on numerical data.

### PCA (Principal Component Analysis)
A statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of linearly uncorrelated principal components.  
This algorithm primarily operates on numerical data.

---

## Note on PCA Location

Although Principal Component Analysis (PCA) is primarily used as a dimensionality reduction technique and is implemented within the preprocessing sub-folder of our main ML package (`src/ml/preprocessing/`), we have included its example notebook within this Unsupervised Learning folder.  
This is done to showcase its functionality alongside other core structure-finding algorithms, making the example self-contained and easy to find.
