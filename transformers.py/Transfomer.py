import Estimator

class Transformer(Estimator.Estimator):
    """
    Base class for all transformers.
    Inherits from the Estimator base class.
    """
    def __init__(self):
        """
        Initializes the Transformer.
        """
        super().__init__()

    def fit(self, X, y=None):
        """
        Fits the transformer on the data.
        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.

        Returns:
            self: Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Transforms the data.
        Parameters:
            X (array-like): Input data.

        Returns:
            array-like: Transformed data.
        """
        return X

    def fit_transform(self, X, y=None):
        """
        Fits the transformer and then transforms the data.
        Parameters:
            X (array-like): Input data.
            y (array-like, optional): Target data.

        Returns:
            array-like: Transformed data.
        """
        return self.fit(X, y).transform(X)
