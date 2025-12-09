import copy
from typing import Any, Dict, List, Optional, Union

import numpy as np
from ml.post_processing.label_post_processing import averageLabel, majorityLabel
from ml.utils._errors_and_warnings._general_error_handling import InvalidSignatureError, _ensure_array_like
from ml.utils.per_sample import _apply_per_sample

class RandomSubspaceEnsemble:
    """
    A General Random Subspace Method (RSM) Ensemble.

    This ensemble improves model robustness and diversity by training base estimators 
    on random subsets of features. It supports heterogeneous models (e.g., mixing 
    Decision Trees with KNNs) via flexible string-based method signatures.

    The ensemble supports:
    1. **Eager Learners:** Models trained during `fit` (e.g., SVM, Decision Trees).
    2. **Lazy Learners:** Models that store training data and compute during `predict` (e.g., KNN).
    3. **Hybrid Tasks:** Can be configured for classification (voting) or regression (averaging).

    Parameters
    ----------
    n_features_in_subset : str, int, or float, default='auto'
        The size of the random feature subspaces.
        - "auto": Uses `int(sqrt(n_features))`.
        - int: Uses the exact integer number of features.
        - float (< 1.0): Uses `int(n_features * float)`.
    task_type : str, default='classification'
        Determines the aggregation strategy.
        - 'classification': Uses `majorityLabel` (Mode).
        - 'regression': Uses `averageLabel` (Mean).

    Attributes
    ----------
    models : List[Dict]
        Internal storage for model instances, signatures, and their assigned feature indices.
    is_fitted : bool
        True if `fit` has been called.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> 
    >>> # 1. Setup Data
    >>> X_train = np.array([[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.9], [0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1]])
    >>> y_train = np.array([0, 1, 0, 0])
    >>> X_test = np.array([[0.2, 0.2, 0.2, 0.2], [0.8, 0.1, 0.1, 0.1]])
    >>> 
    >>> # 2. Initialize Ensemble (Classification with 2 features per model)
    >>> ensemble = RandomSubspaceEnsemble(n_features_in_subset=2, task_type='classification')
    >>> 
    >>> # 3. Add Model
    >>> ensemble.add_model(DecisionTreeClassifier(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)", n_repeats=3)
    >>> 
    >>> # 4. Fit Ensemble
    >>> ensemble.fit(X_train, y_train)
    <...RandomSubspaceEnsemble object at 0x...>
    >>> 
    >>> # 5. Predict (Output will vary slightly due to feature randomization)
    >>> preds = ensemble.predict(X_test)
    >>> len(preds) == len(X_test)
    True
    """

    def __init__(self, n_features_in_subset: Union[str, int, float] = "auto", task_type: str = "classification"):
        self.models: List[Dict[str, Any]] = []
        self.n_features_in_subset = n_features_in_subset
        self.task_type = task_type.lower()
        self.is_fitted = False
        
        if self.task_type not in ['classification', 'regression']:
            raise ValueError("task_type must be 'classification' or 'regression'")

    def add_model(self, 
                  model_init: object, 
                  train_signature: Optional[str], 
                  predict_signature: str, 
                  n_repeats: int = 1):
        """
        Add a base estimator (or multiple copies) to the ensemble.

        Parameters
        ----------
        model_init : object
            An initialized model instance.
        train_signature : str or None
            The method signature for training. 
            - Keywords: `X_TRAIN`, `Y_TRAIN`.
            - Pass `None` or `""` if the model requires no training (pure lazy learner).
        predict_signature : str
            The method signature for prediction.
            - Keywords: `X_TEST` (required), `X_TRAIN` (optional), `Y_TRAIN` (optional).
            - Note: `X_TRAIN`/`Y_TRAIN` allow injecting stored data into lazy learners.
        n_repeats : int, default=1
            Number of independent copies of this model to add. Each copy gets a 
            unique random feature subset.

        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> 
        >>> # Assume `ensemble` object is initialized (e.g., from class examples)
        >>> # Define the necessary variables for the example
        >>> ensemble = RandomSubspaceEnsemble(n_features_in_subset=2, task_type='classification')
        >>> 
        >>> # Add a standard Scikit-learn model
        >>> ensemble.add_model(SVC(gamma='auto'), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
        >>> len(ensemble.models)
        1
        >>> 
        >>> # Add 5 copies of a Logistic Regression model
        >>> from sklearn.linear_model import LogisticRegression
        >>> ensemble.add_model(LogisticRegression(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)", n_repeats=5)
        >>> len(ensemble.models)
        6
        """
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")

        for _ in range(n_repeats):
            self.models.append({
                'model': copy.deepcopy(model_init),
                'train_sig': train_signature,
                'predict_sig': predict_signature,
                'feature_indices': None,    
                'stored_X_train': None,     
                'stored_y_train': None      
            })

    def _execute_signature(self, model: object, signature: str, data_context: Dict[str, Any]):
        """
        Parses and executes a string signature on a model object.
        
        Raises
        ------
        InvalidSignatureError
            If the signature string contains syntax errors, references missing methods,
            or uses variables not present in the data context.
        RuntimeError
            If the model crashes during execution (e.g., inside a called fit/predict method).
        """
        if not signature or signature.strip() == "":
            return None

        local_scope = {
            'model': model,
            **data_context
        }

        full_command = f"model.{signature}"

        try:
            # Execute the signature string securely
            return eval(full_command, {"__builtins__": {}}, local_scope)
        
        except AttributeError as e:
            # Occurs when the method name (e.g. 'fit_magic') does not exist on the model
            model_name = type(model).__name__
            raise InvalidSignatureError(
                f"Method not found on model '{model_name}' using signature '{signature}'. {str(e)}"
            ) from e
            
        except NameError as e:
            # Occurs when a variable in the signature (e.g. 'WRONG_VAR') is not in context
            raise InvalidSignatureError(
                f"Unknown variable in signature '{signature}'. {str(e)}"
            ) from e
            
        except SyntaxError as e:
            # Occurs when the string is not valid Python code (e.g. missing parenthesis)
            raise InvalidSignatureError(
                f"Syntax error in signature '{signature}'. {str(e)}"
            ) from e
            
        except Exception as e:
            # Catch exceptions that occur *inside* the model's method call (e.g., ValueError during fit)
            # and wrap them as RuntimeErrors to distinguish them from simple signature errors.
            raise RuntimeError(f"Model crashed while executing '{full_command}': {e}") from e
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomSubspaceEnsemble':
        """
        Train the ensemble by selecting random subspaces and fitting base models.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        Returns
        -------
        self : RandomSubspaceEnsemble
            The fitted model.

        Examples
        --------
        >>> # Assume `ensemble` object and `X_train`, `y_train` are initialized
        >>> # 1. Define the necessary variables for the example
        >>> ensemble = RandomSubspaceEnsemble(n_features_in_subset=2, task_type='classification')
        >>> X_train = np.array([[0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.9, 0.9], [0.6, 0.1, 0.1, 0.1], [0.1, 0.1, 0.6, 0.1]])
        >>> y_train = np.array([0, 1, 0, 0])
        >>> 
        >>> # 2. Add a base model
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> ensemble.add_model(KNeighborsClassifier(n_neighbors=1), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)")
        >>> 
        >>> # 3. Fit the ensemble
        >>> ensemble.fit(X_train, y_train)
        <...RandomSubspaceEnsemble object at 0x...>
        >>> ensemble.is_fitted
        True
        """
        X = _ensure_array_like(X, name = "X")
        y = _ensure_array_like(y, name = "y")
        
        n_samples, n_total_features = X.shape
        
        # Determine subset size
        if self.n_features_in_subset == "auto":
            n_subset = int(np.sqrt(n_total_features))
        elif isinstance(self.n_features_in_subset, float) and self.n_features_in_subset < 1.0:
            n_subset = int(n_total_features * self.n_features_in_subset)
        else:
            n_subset = int(self.n_features_in_subset)
        
        # Clamp subset size
        n_subset = max(1, min(n_subset, n_total_features))
        
        for model_entry in self.models:
            # 1. Random Subspace Selection
            indices = np.random.choice(n_total_features, n_subset, replace=False)
            model_entry['feature_indices'] = indices
            
            # 2. Slice Data (Projection)
            X_subset = X[:, indices]
            
            # 3. Store Data (Context for Lazy Learners)
            model_entry['stored_X_train'] = X_subset
            model_entry['stored_y_train'] = y
            
            # 4. Train Eager Learners
            if model_entry['train_sig']:
                context = {'X_TRAIN': X_subset, 'Y_TRAIN': y}
                self._execute_signature(model_entry['model'], model_entry['train_sig'], context)
            
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels or values for input samples X by aggregating base model predictions.

        The method first projects the input data into the specific feature subspace 
        assigned to each model. It then executes the model's prediction signature 
        and aggregates the results using either `majorityLabel` (classification) 
        or `averageLabel` (regression).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples for which to generate predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The final aggregated prediction for each input sample.

        Raises
        ------
        RuntimeError
            If the model has not been fitted prior to calling predict.
        ValueError
            If an unknown task type is encountered.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> 
        >>> # --- Setup Data for Regression ---
        >>> X_train = np.array([[10], [20], [30], [40]])
        >>> y_train = np.array([12, 21, 33, 40])
        >>> X_test = np.array([[15], [35]])
        >>> 
        >>> # --- Setup Ensemble for Regression ---
        >>> ensemble_reg = RandomSubspaceEnsemble(n_features_in_subset=1, task_type="regression")
        >>> ensemble_reg.add_model(LinearRegression(), "fit(X_TRAIN, Y_TRAIN)", "predict(X_TEST)", n_repeats=5)
        >>> ensemble_reg.fit(X_train, y_train)
        <...RandomSubspaceEnsemble object at 0x...>
        >>> # --- Predict (Regression) ---
        >>> preds_reg = ensemble_reg.predict(X_test)
        >>> print(preds_reg)
        [16.9... 36.1...]
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict.")
            
        # Ensure X is a NumPy array and pass 'name' argument for validation utility
        X = _ensure_array_like(X, name='X') 

        all_predictions: List[np.ndarray] = []
        
        for model_entry in self.models:
            # Select the specific features for this model's subspace
            indices = model_entry['feature_indices']
            X_subset = X[:, indices]
            
            # Define a wrapper function to execute the prediction signature for a single sample.
            def _predict_wrapper(x_in):
                context: Dict[str, Any] = {
                    'X_TEST': x_in, 
                    'X_TRAIN': model_entry['stored_X_train'], 
                    'Y_TRAIN': model_entry['stored_y_train']
                }
                # Execute the signature string (e.g., 'predict(X_TEST)') using eval
                return self._execute_signature(model_entry['model'], model_entry['predict_sig'], context)

            # Apply the prediction wrapper across all samples in the feature subspace.
            # This utility handles both vectorized and row-wise non-vectorized execution.
            preds = _apply_per_sample(_predict_wrapper, X_subset)
            
            # Ensure the prediction result is a flat 1D array before stacking
            preds = np.atleast_1d(np.asarray(preds).flatten()) 
            all_predictions.append(preds)
            
        # Stack predictions: shape (n_models, n_samples). This prepares for aggregation.
        stacked_preds = np.vstack(all_predictions) 
        
        # --- Aggregation ---
        # Apply the appropriate aggregation function across the models (axis=0) for each sample.
        if self.task_type == "classification":
            # Use majority vote for classification
            final_preds = np.apply_along_axis(majorityLabel, axis=0, arr=stacked_preds)
                
        elif self.task_type == "regression":
            # Use simple averaging for regression
            final_preds = np.apply_along_axis(averageLabel, axis=0, arr=stacked_preds)
        
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}. Must be 'classification' or 'regression'.")
            
        return final_preds