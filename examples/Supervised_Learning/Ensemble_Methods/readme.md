# Ensemble Methods

## Overview

Ensemble methods are meta-algorithms that combine several machine learning techniques into one predictive model in order to decrease variance (bagging), bias (boosting), or improve predictions (stacking). The fundamental idea is that a group of "weak learners" can come together to form a "strong learner" that is more robust and accurate than any individual model.

This module currently implements the Random Subspace Ensemble (RSE) method.

---

## Random Subspace Ensemble (RSE)

The Random Subspace Method (also known as attribute bagging) is an ensemble technique that attempts to reduce the correlation between estimators in an ensemble by training them on random subsets of features.

### The Algorithm

**Initialization:**  
Define the number of base estimators ($N$) and the size of the feature subset ($m < M_{total}$).

**Feature Selection:**  
For each base estimator, randomly select $m$ features from the original feature space.

**Training:**  
Train the base estimator using the full set of training samples (rows), but only the selected feature subset (columns).

**Aggregation:**

- **Classification:** The ensemble predicts the class that receives the most votes (Majority Voting).
- **Regression:** The ensemble predicts the arithmetic mean of the individual model outputs (Averaging).

---

## When to Use It

### ✅ Good For:

- **High-Dimensional Data:**  
  RSE shines when the number of features is very large compared to the number of samples (the "Curse of Dimensionality"). It helps by focusing on different views of the data.

- **Redundant Features:**  
  If your dataset has many features that are correlated or provide similar information, RSE prevents one dominant feature from overshadowing others in every model.

- **Variance Reduction:**  
  Similar to bagging, it helps prevent overfitting by averaging out the errors of individual models.

### ❌ Less Effective For:

- **Irrelevant Features:**  
  If a dataset contains many noise features (garbage inputs), RSE may force some models to train only on noise, leading to poor performance.

- **Low-Dimensional Data:**  
  If you have very few features, there isn't enough diversity to create meaningful subspaces.

---

## Implementation Functionality

The `RandomSubspaceEnsemble` class in this module is designed for maximum flexibility, allowing you to combine heterogeneous models (e.g., mixing Decision Trees with KNNs) and handle non-standard model interfaces.

### Key Features

#### Flexible String Signatures:

The implementation allows you to define exactly how your model's methods should be called using string templates, providing great flexibility. This is handled via the `add_model` method, which accepts `train_signature` and `predict_signature` strings.

- Example: `"fit(X_TRAIN, Y_TRAIN, epochs=10)"`
- Example: `"predict(target=X_TEST, K=5)"`

#### Support for Lazy Learners:

Models like KNN often require access to training data at prediction time. This implementation automatically stores the specific training subset for each model and allows you to inject it during prediction using the `X_TRAIN` and `Y_TRAIN` keywords in the signature.

#### Heterogeneous Ensembles:

You are not restricted to a single base model type. You can add various types of models to the same ensemble, each looking at different feature subspaces.

#### Task Agnostic:

Supports both Classification and Regression tasks.

---

# API Reference

## `__init__(self, n_features_in_subset="auto", task_type="classification")`

Initializes the ensemble configuration.

- **n_features_in_subset:** Determines the size of the feature subspace for each model.  
  Can be an integer (exact count), a float (percentage), or `"auto"` (sqrt of total features).

- **task_type:** Defines the aggregation strategy. Options are `"classification"` (majority vote) or `"regression"` (averaging).

---

## `add_model(self, model_init, train_signature, predict_signature, n_repeats=1)`

Adds a base estimator (or multiple copies) to the ensemble.

- **model_init:** An initialized instance of the base model (e.g., `DecisionTreeClassifier()`).
- **train_signature:** A string defining the training call. Use keywords `X_TRAIN` and `Y_TRAIN`. Pass `None` for lazy learners.
- **predict_signature:** A string defining the prediction call. Use keywords `X_TEST`, and optionally `X_TRAIN`/`Y_TRAIN` for lazy learners.
- **n_repeats:** The number of independent copies of this model configuration to add.

---

## `fit(self, X, y)`

Trains the ensemble.

- **X:** The input training data (array-like).  
- **y:** The target values (array-like).

**Behavior:**  
For every model added, it selects a random subset of features, projects `X` into that subspace, and executes the `train_signature`.

---

## `predict(self, X)`

Generates predictions for the input data.

- **X:** The input data to predict on (array-like).  
- **Returns:** An array of aggregated predictions.

**Behavior:**  
Projects `X` into the specific feature subspace of each model, executes the `predict_signature`, and aggregates the results based on the `task_type`.