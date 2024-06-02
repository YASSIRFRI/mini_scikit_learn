import numpy as np
from .Estimator import Estimator
from .Predictor import Predictor


class DecisionTreeClassifier(Predictor, Estimator):
    
    def __init__(self, max_depth=1000):
        """This is the constructor of the class.
        Parameters:
        max_depth (int): The maximum depth of the tree.
        """
        self.max_depth = max_depth
        self.tree = None
        self.is_fitted = False
        super().__init__(self)
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        self.is_fitted = True
        return self

    def get_params(self):
        return {"max_depth": self.max_depth}

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.array([self._predict_tree(x, self.tree) for x in X])
    
    def score(self, X, y):
        """This method is used to evaluate the model on the test data."""
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

    def _build_tree(self, X, y, depth):
        # Check termination conditions
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.argmax(np.bincount(y))  # Return the majority class index

        # Find the best split
        best_split = self._find_best_split(X, y)

        if best_split is None:
            return np.argmax(np.bincount(y))  # Return the majority class index if no split found

        feature_index, threshold = best_split

        # Split the data
        left_indices = X[:, feature_index] < threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # Recursively build subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return (feature_index, threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
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

                # Calculate entropy for the child nodes
                entropy_left = self._entropy(left_labels)
                entropy_right = self._entropy(right_labels)

                # Calculate weighted average entropy
                weight_left = len(left_labels) / len(y)
                weight_right = len(right_labels) / len(y)
                weighted_avg_entropy = weight_left * entropy_left + weight_right * entropy_right

                # Calculate entropy gain
                entropy_gain = parent_entropy - weighted_avg_entropy

                if entropy_gain > best_entropy_gain:
                    best_entropy_gain = entropy_gain
                    best_split = (feature_index, threshold)

        return best_split

    def _entropy(self, labels):
        class_probabilities = [len(labels[labels == c]) / len(labels) for c in np.unique(labels)]
        entropy = -np.sum(p * np.log2(p) for p in class_probabilities if p > 0)
        return entropy

    def _gini_impurity(self, left_labels, right_labels):
        total_samples = len(left_labels) + len(right_labels)
        p_left = len(left_labels) / total_samples
        p_right = len(right_labels) / total_samples
        gini_left = 1 - sum((np.mean(left_labels == c) ** 2 for c in set(left_labels)))
        gini_right = 1 - sum((np.mean(right_labels == c) ** 2 for c in set(right_labels)))

        gini_impurity = p_left * gini_left + p_right * gini_right
        return gini_impurity

    def _predict_tree(self, x, tree):
        if isinstance(tree, np.int64):  
            return tree
        else:  
            feature_index, threshold, left_subtree, right_subtree = tree
            if x[feature_index] < threshold:
                return self._predict_tree(x, left_subtree)
            else:
                return self._predict_tree(x, right_subtree)
            









class DecisionTreeRegressor(Predictor, Estimator):
    
    def __init__(self, max_depth=1000):
        """This is the constructor of the class.
        Parameters:
        max_depth (int): The maximum depth of the tree.
        """
        self.max_depth = max_depth
        self.tree = None
        self.is_fitted = False
        super().__init__(self)
    
    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)
        self.is_fitted = True
        return self

    def get_params(self):
        return {"max_depth": self.max_depth}

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.array([self._predict_tree(x, self.tree) for x in X])
    
    def score(self, X, y):
        """This method is used to evaluate the model on the test data."""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)  # Mean Squared Error

    def _build_tree(self, X, y, depth):
        # Check termination conditions
        if depth >= self.max_depth or len(y) <= 1:
            return np.mean(y)  # Return the mean value of y

        # Find the best split
        best_split = self._find_best_split(X, y)

        if best_split is None:
            return np.mean(y)  # Return the mean value of y if no split found

        feature_index, threshold = best_split

        # Split the data
        left_indices = X[:, feature_index] < threshold
        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[~left_indices], y[~left_indices]

        # Recursively build subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return (feature_index, threshold, left_subtree, right_subtree)

    def _find_best_split(self, X, y):
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

                # Calculate mean squared error for the child nodes
                mse_left = self._mean_squared_error(left_labels)
                mse_right = self._mean_squared_error(right_labels)

                # Calculate weighted average mean squared error
                weight_left = len(left_labels) / len(y)
                weight_right = len(right_labels) / len(y)
                weighted_avg_mse = weight_left * mse_left + weight_right * mse_right

                # Calculate mse gain
                mse_gain = parent_mse - weighted_avg_mse

                if mse_gain > best_mse_gain:
                    best_mse_gain = mse_gain
                    best_split = (feature_index, threshold)

        return best_split

    def _mean_squared_error(self, labels):
        mean = np.mean(labels)
        mse = np.mean((labels - mean) ** 2)
        return mse

    def _predict_tree(self, x, tree):
        if isinstance(tree, float):  # Leaf node
            return tree
        else:  # Decision node
            feature_index, threshold, left_subtree, right_subtree = tree
            if x[feature_index] < threshold:
                return self._predict_tree(x, left_subtree)
            else:
                return self._predict_tree(x, right_subtree)

