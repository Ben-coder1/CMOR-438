# Examples: Demonstrating the ML Package

This folder, `examples/`, contains a collection of Jupyter notebooks designed to demonstrate how to effectively use the algorithms and components within our custom machine learning package, located in `src/ml/`.

---

## Key Features

### Hands-on Use Cases
Each notebook provides a step-by-step example of implementing a specific ML algorithm—such as Linear Regression, K-Nearest Neighbors, or a Decision Tree—for a task like classification or regression.

### Detailed Explanations
Alongside the executable code, these notebooks include detailed readmes that explain the purpose of the code, the function of the ML algorithm, and the usage of the specific classes/methods imported from the `src/ml/` package.

### Self-Contained Notebooks
To ensure each example is fully functional and easy to run in isolation (with the exception of importing functions from the `src/ml/` package), many notebooks use the same dataset. For instance, the notebooks for `LinearRegression.ipynb` and `DecisionTreeRegressor.ipynb` might both use the same housing data. This means the underlying data file may be intentionally duplicated across multiple example notebooks. This design choice prioritizes the self-contained nature of each demonstration over strict data non-redundancy.

---

## How to Use

1. Ensure you have the main `src/ml/` package installed or properly configured in your environment.
2. Open any of the `.ipynb` files in this directory.
3. Run the cells sequentially to see the ML algorithm in action, train a model, and evaluate its results.