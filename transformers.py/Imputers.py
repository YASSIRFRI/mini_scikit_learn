from Transfomer import Transformer


class Imputer(Transformer):
    """Base class for imputers."""
    pass



class SimpleImputer(Imputer):
    """Simple imputer."""
    def __init__(self):
        super().__init__()
        self.fill_value = None

    def fit(self, X, y=None, strategy='median'):
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
        return np.where(pd.isnull(X), self.fill_value, X)
    
    
class IterativeImputer(Imputer):
    """Iterative imputer."""
    def __init__(self):
        super().__init__()
        self.estimator = None

    def fit(self, X, y=None, estimator=None):
        if estimator is None:
            estimator = LinearRegression()
        self.estimator = estimator
        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        return self.estimator.predict(X)
    


class KNNImputer(Imputer):
    """KNN imputer."""
    def __init__(self):
        super().__init__()
        self.k = None

    def fit(self, X, y=None, k=5):
        self.k = k
        return self
    
    def transform(self, X):
        return KNNImputer(n_neighbors=self.k).fit_transform(X)
    
    



