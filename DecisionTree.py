import numpy as np
from Estimator import Estimator
from Predictor import Predictor


class DecisionTreeClassifier(Predictor, Estimator):
    
    def __init__(self, max_depth=1000):
        """Initialize the classifier."""
        self.max_depth = max_depth
        self.tree = None
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.tree = self._build_tree(X, y, depth=0)
        self.is_fitted = True
        return self

    def get_params(self):
        """
        Get parameters for this estimator.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return {"max_depth": self.max_depth}

    def predict(self, X):
        """
        Predict class for X.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).

        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data of shape (n_samples, n_features).
        y : numpy.ndarray
            True labels for X.

        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def _build_tree(self, X, y, depth):
        """
        Build the decision tree recursively.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).
        depth : int
            Current depth of the tree.

        Returns:
        --------
        tree : tuple or int
            The root of the decision tree or the majority class index.
        """
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))  # Return the majority class index

        best_split = self._find_best_split(X, y)

        if best_split is None:
            return np.argmax(np.bincount(y))  # Return the majority class index if no split found

        feature_index, threshold = best_split

        left_indices = X[:, feature_index] < threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return (feature_index, threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns:
        --------
        best_split : tuple or None
            The best feature index and threshold to split the data.
        """
        best_entropy_gain = float('-inf')
        best_split = None
        n_features = X.shape[1]
        parent_entropy = self._entropy(y)

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                left_labels = y[left_indices]
                right_labels = y[~left_indices]

                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue  # Skip if one of the child nodes is empty

                entropy_left = self._entropy(left_labels)
                entropy_right = self._entropy(right_labels)

                weight_left = len(left_labels) / len(y)
                weight_right = len(right_labels) / len(y)
                weighted_avg_entropy = weight_left * entropy_left + weight_right * entropy_right

                entropy_gain = parent_entropy - weighted_avg_entropy

                if entropy_gain > best_entropy_gain:
                    best_entropy_gain = entropy_gain
                    best_split = (feature_index, threshold)

        return best_split

    def _entropy(self, labels):
        """
        Calculate the entropy of a label array.

        Parameters:
        -----------
        labels : numpy.ndarray
            Array of labels.

        Returns:
        --------
        entropy : float
            The entropy of the label distribution.
        """
        class_probabilities = [len(labels[labels == c]) / len(labels) for c in np.unique(labels)]
        entropy = -np.sum(p * np.log2(p) for p in class_probabilities if p > 0)
        return entropy

    def _predict_tree(self, x, tree):
        """
        Predict class for a single sample using the decision tree.

        Parameters:
        -----------
        x : numpy.ndarray
            A single input sample.
        tree : tuple or int
            The decision tree or the class index.

        Returns:
        --------
        prediction : int
            Predicted class label.
        """
        if isinstance(tree, np.int64):  # Leaf node
            return tree
        else:  # Decision node
            feature_index, threshold, left_subtree, right_subtree = tree
            if x[feature_index] < threshold:
                return self._predict_tree(x, left_subtree)
            else:
                return self._predict_tree(x, right_subtree)

class DecisionTreeRegressor(Predictor, Estimator):
    
    def __init__(self, max_depth=1000):
        """Initialize the regressor."""
        self.max_depth = max_depth
        self.tree = None
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.tree = self._build_tree(X, y, depth=0)
        self.is_fitted = True
        return self

    def get_params(self):
        """
        Get parameters for this estimator.

        Returns:
        --------
        params : dict
            Parameter names mapped to their values.
        """
        return {"max_depth": self.max_depth}

    def predict(self, X):
        """
        Predict target for X.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).

        Returns:
        --------
        y_pred : numpy.ndarray
            Predicted target values.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def score(self, X, y):
        """
        Returns the mean squared error on the given test data and labels.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data of shape (n_samples, n_features).
        y : numpy.ndarray
            True values for X.

        Returns:
        --------
        score : float
            Mean squared error of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)  # Mean Squared Error

    def _build_tree(self, X, y, depth):
        """
        Build the decision tree recursively.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).
        depth : int
            Current depth of the tree.

        Returns:
        --------
        tree : tuple or float
            The root of the decision tree or the mean value of y.
        """
        if depth >= self.max_depth or len(y) <= 1:
            return np.mean(y)  # Return the mean value of y

        best_split = self._find_best_split(X, y)

        if best_split is None:
            return np.mean(y)  # Return the mean value of y if no split found

        feature_index, threshold = best_split

        left_indices = X[:, feature_index] < threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~right_indices]

        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return (feature_index, threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
        """
        Find the best feature and threshold to split the data.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features).
        y : numpy.ndarray
            Target values of shape (n_samples,).

        Returns:
        --------
        best_split : tuple or None
            The best feature index and threshold to split the data.
        """
        best_mse_gain = float('-inf')
        best_split = None
        n_features = X.shape[1]
        parent_mse = self._mean_squared_error(y)

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] < threshold
                left_labels = y[left_indices]
                right_labels = y[~left_indices]

                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue  # Skip if one of the child nodes is empty

                mse_left = self._mean_squared_error(left_labels)
                mse_right = self._mean_squared_error(right_labels)

                weight_left = len(left_labels) / len(y)
                weight_right = len(right_labels) / len(y)
                weighted_avg_mse = weight_left * mse_left + weight_right * mse_right

                mse_gain = parent_mse - weighted_avg_mse

                if mse_gain > best_mse_gain:
                    best_mse_gain = mse_gain
                    best_split = (feature_index, threshold)

        return best_split

    def _mean_squared_error(self, labels):
        """
        Calculate the mean squared error of a label array.

        Parameters:
        -----------
        labels : numpy.ndarray
            Array of labels.

        Returns:
        --------
        mse : float
            The mean squared error of the label distribution.
        """
        mean = np.mean(labels)
        mse = np.mean((labels - mean) ** 2)
        return mse

    def _predict_tree(self, x, tree):
        """
        Predict target for a single sample using the decision tree.

        Parameters:
        -----------
        x : numpy.ndarray
            A single input sample.
        tree : tuple or float
            The decision tree or the mean value of y.

        Returns:
        --------
        prediction : float
            Predicted target value.
        """
        if isinstance(tree, float):  # Leaf node
            return tree
        else:  # Decision node
            feature_index, threshold, left_subtree, right_subtree = tree
            if x[feature_index] < threshold:
                return self._predict_tree(x, left_subtree)
            else:
                return self._predict_tree(x, right_subtree)
