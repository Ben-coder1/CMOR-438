
import numpy as np
from ml.utils._errors_and_warnings._general_error_handling import (
    _ensure_numeric_array, 
    _ensure_no_nan, 
    _ensure_non_empty
)
from ml.metrics_and_evaluations.evaluation.supervised.loss_functions import MSE

class LinearRegression:
    """
    Ordinary Least Squares (OLS) Linear Regression.
    
    Solves the linear equation y = Xw + b using the closed-form Normal Equation:
    w = (X^T W X)^-1 X^T W y
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # Equation: y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.coef_
    array([1., 2.])
    >>> # Cast to float to handle potential numpy array wrappers
    >>> float(np.round(reg.intercept_, 2))
    3.0
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """
        Fit linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. Useful for handling outliers 
            or heteroscedastic data.

        Returns
        -------
        self : returns an instance of self.

        Examples
        --------
        >>> model = LinearRegression()
        >>> X = np.array([[1], [2], [3]])
        >>> y = np.array([2, 4, 6]) # y = 2x
        >>> model.fit(X, y)
        <...LinearRegression object at ...>
        >>> model.coef_
        array([2.])
        
        >>> # Using Sample Weights (Ignoring an outlier)
        >>> X_outlier = np.array([[1], [2], [10]])
        >>> y_outlier = np.array([2, 4, 100]) # Last point is noise
        >>> weights = np.array([1, 1, 0.01]) # Downweight the noise
        >>> model.fit(X_outlier, y_outlier, sample_weight=weights)
        <...LinearRegression object at ...>
        """
        # 1. Validation
        X = _ensure_numeric_array(X, name="X", ndim=2)
        y = _ensure_numeric_array(y, name="y")
        _ensure_no_nan(X, name="X")
        _ensure_no_nan(y, name="y")
        _ensure_non_empty(X, name="X")

        # Ensure y is 2D for consistency with the Loss functions
        original_y_ndim = y.ndim
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch: X has {X.shape[0]} samples, y has {y.shape[0]}.")

        # 2. Handle Weights
        if sample_weight is not None:
            sample_weight = _ensure_numeric_array(sample_weight, name="sample_weight")
            if sample_weight.shape[0] != y.shape[0]:
                raise ValueError("sample_weight must have same length as y.")
            W = np.diag(sample_weight)
        else:
            W = np.eye(X.shape[0])

        # 3. Handle Intercept
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X_train = np.hstack((ones, X))
        else:
            X_train = X

        # 4. Normal Equation
        XT_W = X_train.T @ W
        lhs = XT_W @ X_train
        rhs = XT_W @ y
        
        try:
            theta = np.linalg.pinv(lhs) @ rhs
        except np.linalg.LinAlgError:
            raise ValueError("Singular matrix: Linear regression cannot be solved.")

        # 5. Store Parameters
        if self.fit_intercept:
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta

        # Flatten coef/intercept if y was originally 1D for cleaner API usage
        if original_y_ndim == 1 and self.coef_.shape[1] == 1:
            self.coef_ = self.coef_.flatten()
            self.intercept_ = self.intercept_.flatten() if isinstance(self.intercept_, np.ndarray) else self.intercept_

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples,)
            Returns predicted values.

        Examples
        --------
        >>> model = LinearRegression()
        >>> model.fit(np.array([[1], [2]]), np.array([3, 5])) # y = 2x + 1
        <...LinearRegression object at ...>
        >>> model.predict(np.array([[3], [4]]))
        array([7., 9.])
        """
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")

        X = _ensure_numeric_array(X, name="X", ndim=2)
        _ensure_no_nan(X, name="X")
        
        return X @ self.coef_ + self.intercept_

    def mean_squared_error(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates MSE for the model on a given dataset (X, y_true).
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y_true : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        float
            Mean squared error regression loss.

        Examples
        --------
        >>> model = LinearRegression()
        >>> X_train = np.array([[1], [2]])
        >>> y_train = np.array([2, 4])
        >>> model.fit(X_train, y_train)
        <...LinearRegression object at ...>
        >>> # Test on same data (should be 0 error for perfect linear fit)
        >>> mse = model.mean_squared_error(X_train, y_train)
        >>> bool(np.isclose(mse, 0.0))
        True
        >>> # Test on slightly noisy data
        >>> X_test = np.array([[3]])
        >>> y_test_noisy = np.array([6.5]) # Prediction should be 6
        >>> mse = model.mean_squared_error(X_test, y_test_noisy)
        >>> bool(np.isclose(mse, 0.25))
        True
        """
        # Generate predictions
        y_pred = self.predict(X)
        
        # Ensure 2D shape (n_samples, n_targets) for the Loss function
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)

        # Call .mean_loss() to get the single scalar average over all samples
        return MSE.mean_loss(y_true, y_pred)