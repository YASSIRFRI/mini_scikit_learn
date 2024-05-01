import numpy as np
import Estimator
import Predictor




class DecisionTreeClassifier(Predictor.Predictor, Estimator.Estimator):
    
    def __init__(self, max_depth=5):
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

    def _build_tree(self, X, y, depth):
        if len(y) == 0:
            return {"class": None}
        elif self.max_depth is not None and depth == self.max_depth:
            return {"class": np.argmax(np.bincount(y))}
        elif len(np.unique(y)) == 1:
            return {"class": y[0]}
        else:
            best_feature, best_threshold = self._best_split(X, y)
            if best_feature is None:
                return {"class": np.argmax(np.bincount(y))}
            left_indices = X[:, best_feature] < best_threshold
            right_indices = X[:, best_feature] >= best_threshold
            left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
            right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
            return {"feature": best_feature, "threshold": best_threshold, "left": left_tree, "right": right_tree}

    def _best_split(self, X, y):
        best_gini = float("inf")
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                left_gini = self._gini_impurity(y[left_indices])
                right_gini = self._gini_impurity(y[right_indices])
                gini = left_gini + right_gini
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _gini_impurity(self, y):
        if len(y) == 0:
            return 0
        else:
            probabilities = np.bincount(y) / len(y)
            return 1 - np.sum(probabilities ** 2)

    def _predict_tree(self, x, tree):
        if "class" in tree:
            return tree["class"]
        else:
            if x[tree["feature"]] < tree["threshold"]:
                return self._predict_tree(x, tree["left"])
            else:
                return self._predict_tree(x, tree["right"])