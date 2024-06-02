import numpy as np
from . import Estimator
from . import Predictor
from .DecisionTree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

class RandomForestClassifier(Predictor.Predictor, Estimator.Estimator):
    """
    Random Forest Classifier.

    Parameters:
    -----------
    n_estimators : int, optional (default=100)
        The number of trees in the forest.
    max_depth : int, optional (default=1000)
        The maximum depth of the trees.
    max_features : str or int, optional (default='sqrt')
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If 'sqrt', then consider `sqrt(n_features)` features at each split.
        - If 'log2', then consider `log2(n_features)` features at each split.

    Attributes:
    -----------
    n_estimators : int
        The number of trees in the forest.
    max_depth : int
        The maximum depth of the trees.
    max_features : str or int
        The number of features to consider when looking for the best split.
    trees : list
        List of decision trees in the forest.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """

    def __init__(self, n_estimators=100, max_depth=1000, max_features='sqrt'):
        """Initialize the Random Forest Classifier."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        """
        Fit the random forest model.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Training labels.
        """
        self.trees = []
        n_samples, n_features = X.shape
        
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = self.max_features
        
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree._find_best_split = self._get_find_best_split(max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.is_fitted = True
        return self

    def _get_find_best_split(self, max_features):
        """Return a function that finds the best split considering `max_features` features."""
        def _find_best_split(X, y):
            best_entropy_gain = float('-inf')
            best_split = None
            n_features = X.shape[1]
            parent_entropy = self._entropy(y)
            
            features = np.random.choice(n_features, max_features, replace=False)
            
            for feature_index in features:
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    left_indices = X[:, feature_index] < threshold
                    left_labels = y[left_indices]
                    right_labels = y[~left_indices]
                    
                    if len(left_labels) == 0 or len(right_labels) == 0:
                        continue
                    
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
        
        return _find_best_split

    def _entropy(self, labels):
        """Calculate the entropy of a set of labels."""
        class_probabilities = [len(labels[labels == c]) / len(labels) for c in np.unique(labels)]
        entropy = -np.sum(p * np.log2(p) for p in class_probabilities if p > 0)
        return entropy

    def predict(self, X):
        """
        Predict class for X.

        Parameters:
        -----------
        X : np.ndarray
            Test data.

        Returns:
        --------
        np.ndarray
            Predicted classes.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        majority_vote = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
        return majority_vote
    
    def score(self, X, y):
        """
        Evaluate the model on the test data.

        Parameters:
        -----------
        X : np.ndarray
            Test data.
        y : np.ndarray
            True labels.

        Returns:
        --------
        float
            Accuracy score.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

class RandomForestRegressor:
    """
    Random Forest Regressor.

    Parameters:
    -----------
    n_estimators : int, optional (default=100)
        The number of trees in the forest.
    max_depth : int, optional (default=1000)
        The maximum depth of the trees.
    max_features : str or int, optional (default='sqrt')
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If 'sqrt', then consider `sqrt(n_features)` features at each split.
        - If 'log2', then consider `log2(n_features)` features at each split.

    Attributes:
    -----------
    n_estimators : int
        The number of trees in the forest.
    max_depth : int
        The maximum depth of the trees.
    max_features : str or int
        The number of features to consider when looking for the best split.
    trees : list
        List of decision trees in the forest.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """

    def __init__(self, n_estimators=100, max_depth=1000, max_features='sqrt'):
        """Initialize the Random Forest Regressor."""
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.is_fitted = False

    def fit(self, X, y):
        """
        Fit the random forest model.

        Parameters:
        -----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Training labels.
        """
        self.trees = []
        n_samples, n_features = X.shape
        
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = self.max_features
        
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree._find_best_split = self._get_find_best_split(max_features)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.is_fitted = True
        return self

    def _get_find_best_split(self, max_features):
        """Return a function that finds the best split considering `max_features` features."""
        def _find_best_split(X, y):
            best_variance_reduction = float('-inf')
            best_split = None
            n_features = X.shape[1]
            parent_variance = np.var(y)
            
            features = np.random.choice(n_features, max_features, replace=False)
            
            for feature_index in features:
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    left_indices = X[:, feature_index] < threshold
                    left_labels = y[left_indices]
                    right_labels = y[~left_indices]
                    
                    if len(left_labels) == 0 or len(right_labels) == 0:
                        continue
                    
                    variance_left = np.var(left_labels)
                    variance_right = np.var(right_labels)
                    
                    weight_left = len(left_labels) / len(y)
                    weight_right = len(right_labels) / len(y)
                    weighted_avg_variance = weight_left * variance_left + weight_right * variance_right
                    
                    variance_reduction = parent_variance - weighted_avg_variance
                    
                    if variance_reduction > best_variance_reduction:
                        best_variance_reduction = variance_reduction
                        best_split = (feature_index, threshold)
            
            return best_split
        
        return _find_best_split

    def predict(self, X):
        """
        Predict values for X.

        Parameters:
        -----------
        X : np.ndarray
            Test data.

        Returns:
        --------
        np.ndarray
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        mean_predictions = np.mean(tree_predictions, axis=0)
        return mean_predictions
    
    def score(self, X, y):
        """
        Evaluate the model on the test data.

        Parameters:
        -----------
        X : np.ndarray
            Test data.
        y : np.ndarray
            True values.

        Returns:
        --------
        float
            Negative mean squared error.
        """
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)
    
    
    def set_params(self, **params):
        """Set the parameters of the model."""
        for param, value in params.items():
            setattr(self, param, value)
        return self
