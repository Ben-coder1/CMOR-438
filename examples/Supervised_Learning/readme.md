# Supervised Learning

Supervised learning is a category of machine learning in which a model is trained using **labeled data** — each input example comes with a correct output value. The objective is to learn a mapping from inputs to outputs so the model can make accurate predictions on new, unseen data.

## When to Use Supervised Learning

Use supervised learning when:

- You want to **make predictions** about outcomes based on historical data.
- Your task is one of:
  - **Regression** — predicting continuous numeric values (e.g., house prices, temperatures), or  
  - **Classification** — predicting discrete categories or labels (e.g., spam vs. not spam, image labels).
- You have access to **training data with correct values (labels)**.  
  Supervised learning requires labeled examples during training; without good labels the model cannot learn the correct mapping.

## Models Included in This Package

This package includes implementations of common supervised learning algorithms:

- **Decision Trees** — Tree-structured models that split data by feature values. Useful for both regression and classification.
- **K-Nearest Neighbors (KNN)** — Instance-based method that predicts based on the labels of the nearest training samples.
- **Linear Regression** — Predicts continuous outputs assuming a linear relationship between features and target.
- **Logistic Regression** — A linear model for binary (and extended for multiclass) classification using a logistic link.
- **Multilayer Perceptron (MLP)** — Feedforward neural network capable of modeling non-linear relationships for regression and classification.
- **Perceptron** — A classic linear classifier suited to linearly separable problems.
- **Regression Trees** — A type of Decision Tree specifically used for regression tasks. The model recursively partitions the feature space into a set of non-overlapping regions, and for any new data point falling into a specific region, the prediction is the average (or other measure like the median) of the target values of the training points in that region.

## Quick Start

1. **Prepare data**: collect features (X) and correct target values/labels (y).  
2. **Choose a model**: pick regression if the target is continuous; classification if it is categorical.  
3. **Train**: fit the chosen model on your labeled training data.  
4. **Validate**: evaluate model performance on a validation or test set (e.g., accuracy, precision/recall for classification; MSE, MAE for regression).  
5. **Predict**: use the trained model to make predictions on new, unseen inputs.