from .. import Estimator
import numpy as np

class Transformer(Estimator.Estimator):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    


class MinMaxScaler(Transformer):
    def __init__(self):
        super().__init__()
        self.min = None
        self.max = None

    def fit(self, X, y=None):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)
    

class StandardScaler(Transformer):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean

