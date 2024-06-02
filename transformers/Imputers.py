from .Transfomer import Transformer


class Imputer(Transformer):
    """Base class for imputers."""
    pass

class SimpleImputer(Imputer):
    """Simple imputer."""
    def __init__(self):
        """
        Initializes the SimpleImputer.
        """
        super().__init__()
        self.fill_value = None

    def fit(self, X, y=None, strategy='median'):
        """
        Fits the imputer on the data using the specified strategy.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.
            strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent').

        Returns:
            self: Returns the instance itself.
        """
        if strategy == 'mean':
            self.fill_value = np.nanmean(X)
        elif strategy == 'median':
            self.fill_value = np.nanmedian(X)
        elif strategy == 'most_frequent':
            self.fill_value = pd.Series(X).mode().iloc[0]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted imputer.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return np.where(pd.isnull(X), self.fill_value, X)
    
class IterativeImputer(Imputer):
    """Iterative imputer."""
    def __init__(self):
        """
        Initializes the IterativeImputer.
        """
        super().__init__()
        self.estimator = None

    def fit(self, X, y=None, estimator=None):
        """
        Fits the imputer on the data using the specified estimator.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.
            estimator (object): Estimator to use for imputation.

        Returns:
            self: Returns the instance itself.
        """
        if estimator is None:
            estimator = LinearRegression()
        self.estimator = estimator
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        """
        Transforms the data using the fitted estimator.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return self.estimator.predict(X)

class KNNImputer(Imputer):
    """KNN imputer."""
    def __init__(self):
        """
        Initializes the KNNImputer.
        """
        super().__init__()
        self.k = None

    def fit(self, X, y=None, k=5):
        """
        Fits the imputer on the data using k-nearest neighbors.

        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.
            k (int): Number of neighbors to use for imputation.

        Returns:
            self: Returns the instance itself.
        """
        self.k = k
        return self
    
    def transform(self, X):
        """
        Transforms the data using k-nearest neighbors.

        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return SklearnKNNImputer(n_neighbors=self.k).fit_transform(X)
