from ml.metrics import EuclideanDistance
from ml.post_process import majorityLabel, averageLabel

class KNN:
    def __init__(self, X=None, y=None):
        """
        Initializes the KNN predictor with optional training data.

        Parameters:
            X (list, optional): List of feature vectors.
            y (list, optional): Corresponding labels.
        """
        self.X = X
        self.y = y

    def find_neighbors(self, target, X, y=None, K=5, dist=None):
        """
        Finds the K nearest neighbors to the target.

        Parameters:
            target: The input vector to compare against.
            X (list): Feature vectors.
            y (list, optional): Corresponding labels. If None, labels will be None.
            K (int): Number of neighbors.
            dist (callable): Distance function.

        Returns:
            List of (x_i, label, distance) tuples for the K closest points.

        Raises:
            ValueError: If target is None, K is too large, or lengths mismatch.
            TypeError: If K is not int or dist is not callable.
        """
        if target is None:
            raise ValueError("Target vector must be provided.")
        if not isinstance(K, int) or K <= 0:
            raise TypeError("K must be a positive integer.")
        if dist is None:
            dist = EuclideanDistance
        if not callable(dist):
            raise TypeError("dist must be a callable function.")
        if not isinstance(X, list):
            raise TypeError("X must be a list.")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must be of the same length.")
        if K > len(X):
            raise ValueError(f"K={K} is too large for dataset of size {len(X)}.")

        distances = []
        for i, x_i in enumerate(X):
            label = y[i] if y is not None else None
            d = dist(target, x_i)
            distances.append((x_i, label, d))

        distances.sort(key=lambda x: x[2])
        return distances[:K]


    def predict(self, target, classify=True, K=5, X=None, y=None, dist=None):
        """
        Predicts a label for the target vector.

        Parameters:
            target: The input vector to classify or regress.
            classify (bool): If True, use majority label; else use average label.
            K (int): Number of neighbors to consider.
            X (list, optional): Feature vectors. Defaults to stored X.
            y (list, optional): Labels. Defaults to stored y.
            dist (callable, optional): Distance function. Defaults to EuclideanDistance.

        Returns:
            Predicted label (str or float).
        """
        if not isinstance(classify, bool):
            raise TypeError("classify must be a boolean.")
        if dist is None:
            dist = EuclideanDistance
        X = X if X is not None else self.X
        y = y if y is not None else self.y
        if X is None or y is None:
            raise ValueError("X and y must be provided either in init or in predict.")
        if not classify and not all(isinstance(label, (int, float)) for label in y):
            raise TypeError("All labels must be numeric for regression.")

        neighbors = self.find_neighbors(target, X, y, K, dist)
        labels = [label for _, label, _ in neighbors]
        return majorityLabel(labels) if classify else averageLabel(labels)

    def error(self, X_test, y_test, K=5, dist=None, classify=True, X_train=None, y_train=None):
        """
        Computes prediction error on test data using stored or provided training data.

        Parameters:
            X_test (list): Test feature vectors.
            y_test (list): True labels for test data.
            K (int): Number of neighbors to consider.
            dist (callable, optional): Distance function. Defaults to EuclideanDistance.
            classify (bool): If True, computes classification error; else regression error.
            X_train (list, optional): Training feature vectors. Defaults to stored X.
            y_train (list, optional): Training labels. Defaults to stored y.

        Returns:
            float: Mean error over the test set (misclassification rate or mean absolute error).
        """
        X_train = X_train if X_train is not None else self.X
        y_train = y_train if y_train is not None else self.y
        if X_train is None or y_train is None:
            raise ValueError("Training data must be provided either in init or in error.")
        if len(X_test) != len(y_test):
            raise ValueError("X_test and y_test must be the same length.")
        if dist is None:
            dist = EuclideanDistance

        errors = 0
        for x_test, y_true in zip(X_test, y_test):
            y_pred = self.predict(x_test, classify=classify, K=K, X=X_train, y=y_train, dist=dist)
            if classify:
                errors += int(y_pred != y_true)
            else:
                errors += abs(y_pred - y_true)

        return errors / len(X_test)
