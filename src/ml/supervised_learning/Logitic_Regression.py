
import numpy as np
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array, 
    _ensure_no_nan, 
    _ensure_non_empty
)
from ml.utils.activations import sigmoid

class LogisticRegression:
    """
    ⭐ Logistic Regression Classifier using Gradient Descent (Binary Classification).

    This model estimates the probability of a binary outcome (0 or 1) by fitting
    a linear function to the data and then applying the sigmoid function to map
    the result to a probability between 0 and 1. The model is trained by minimizing
    the Binary Cross-Entropy loss via Gradient Descent.

    Parameters
    ----------
    learning_rate : float, optional
        The step size taken during each iteration of gradient descent (default is 0.01).
    n_iterations : int, optional
        The number of passes over the entire training set (default is 1000).
    fit_intercept : bool, optional
        Whether to calculate an intercept (bias) term for the model (default is True).

    Attributes
    ----------
    coef_ : np.ndarray
        The learned coefficients (weights) for the features.
    intercept_ : float or None
        The learned intercept (bias) term.
    """
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, fit_intercept: bool = True):
        """
        Initializes the Logistic Regression model parameters.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the logistic model to the training data using Gradient Descent.

        The model iteratively updates the weights and intercept to minimize the
        Binary Cross-Entropy loss.

        Parameters
        ----------
        X : np.ndarray
            Training feature data, shape (n_samples, n_features).
        y : np.ndarray
            Target labels, shape (n_samples,), must contain only 0s and 1s.

        Returns
        -------
        self : LogisticRegression
            Returns the fitted model instance.

        Raises
        ------
        ValueError
            If the target array 'y' contains values other than 0 or 1.

        Examples
        --------
        >>> X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        >>> y_train = np.array([0, 0, 1, 1])
        >>> model = LogisticRegression(learning_rate=0.1, n_iterations=100)
        >>> model.fit(X_train, y_train)
        <...LogisticRegression object at ...>
        """
        # 1. Validation and Conversion (using your utilities)
        X = _ensure_numeric_array(X, name="X", ndim=2)
        y = _ensure_numeric_array(y, name="y", ndim=1)
        _ensure_no_nan(X, name="X")
        _ensure_no_nan(y, name="y")
        _ensure_non_empty(X, name="X")
        
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Target variable 'y' must contain only binary labels (0 or 1).")
        
        n_samples, n_features = X.shape
        
        # 2. Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0 if self.fit_intercept else None

        # 3. Gradient Descent Optimization Loop
        for _ in range(self.n_iterations):
            # Calculate linear output (z = Xw + b)
            linear_output = X @ self.coef_
            if self.fit_intercept:
                linear_output += self.intercept_
            
            # Apply sigmoid activation to get probability (p_hat)
            y_predicted_prob = sigmoid(linear_output)
            
            # Calculate error and gradient of the loss (p_hat - y)
            error = y_predicted_prob - y
            
            # Update coefficients (w)
            d_w = (X.T @ error) / n_samples
            self.coef_ -= self.learning_rate * d_w
            
            # Update intercept (b)
            if self.fit_intercept:
                d_b = np.sum(error) / n_samples
                self.intercept_ -= self.learning_rate * d_b

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculates the probability P(y=1 | X).

        This is the raw probability output before classification based on a threshold.

        Parameters
        ----------
        X : np.ndarray
            Feature data to predict on, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Probability array, shape (n_samples,), with values in [0, 1].

        Examples
        --------
        >>> # Setup: Create and fit a simple toy model
        >>> X_train = np.array([[1.0], [2.0], [8.0], [9.0]])
        >>> y_train = np.array([0, 0, 1, 1])
        >>> model = LogisticRegression(learning_rate=0.5, n_iterations=100)
        >>> model.fit(X_train, y_train)
        <...LogisticRegression object at ...>
        
        >>> X_test = np.array([[1.5], [5.0], [9.9]])
        >>> model.predict_proba(X_test).shape
        (3,)
        >>> np.all((model.predict_proba(X_test) >= 0) & (model.predict_proba(X_test) <= 1))
        np.True_  
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")
            
        linear_output = X @ self.coef_
        if self.fit_intercept:
            linear_output += self.intercept_
            
        return sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predicts the class label (0 or 1) based on a probability threshold.

        Parameters
        ----------
        X : np.ndarray
            Feature data to predict on, shape (n_samples, n_features).
        threshold : float, optional
            The probability cutoff for classifying a sample as 1 (default is 0.5).

        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,), containing only 0s and 1s.

        Examples
        --------
        >>> # Setup: Create and fit a simple toy model
        >>> X_train = np.array([[1.0], [2.0], [8.0], [9.0]])
        >>> y_train = np.array([0, 0, 1, 1])
        >>> model = LogisticRegression(learning_rate=0.5, n_iterations=100)
        >>> model.fit(X_train, y_train)
        <...LogisticRegression object at ...>
        
        >>> X_test = np.array([[1.5], [5.0]]) # Test near boundary (5.0) and far (1.5)
        >>> model.predict(X_test, threshold=0.5)
        array([0, 1])
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    