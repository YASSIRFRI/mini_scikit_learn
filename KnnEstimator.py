import numpy as np
import Estimator
import Predictor

class KnnEstimator(Predictor.Predictor, Estimator.Estimator):
    """
    K-Nearest Neighbors Estimator.

    Parameters:
    -----------
    n_neighbors : int, optional (default=5)
        The number of neighbors to consider for prediction.
    p : int, optional (default=2)
        The order of the Minkowski distance.

    Attributes:
    -----------
    n_neighbors : int
        The number of neighbors to consider for prediction.
    p : int
        The order of the Minkowski distance.
    X_train : numpy.ndarray or None
        The training data.
    y_train : numpy.ndarray or None
        The target values.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """
    
    def __init__(self, n_neighbors=5, p=2):
        """Initialize the KNN estimator."""
        self.n_neighbors = n_neighbors
        self.p = p
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """
        Train the model on the training data.

        Parameters:
        -----------
        X : numpy.ndarray
            The training data.
        y : numpy.ndarray
            The target values.

        Returns:
        --------
        self : object
            The trained model.
        """
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self
    
    def get_params(self):
        """
        Get the parameters of the model.

        Returns:
        --------
        params : dict
            The parameters of the model.
        """
        return {"n_neighbors": self.n_neighbors, "p": self.p}
    
    def predict(self, X):
        """
        Make predictions on the test data.

        Parameters:
        -----------
        X : numpy.ndarray
            The test data.

        Returns:
        --------
        y_pred : numpy.ndarray
            The predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            distances = np.linalg.norm(self.X_train - X[i], ord=self.p, axis=1)
            nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
            y_pred[i] = np.mean(self.y_train[nearest_neighbors])
        return y_pred
    
    def score(self, X, y):
        """
        Evaluate the model on the test data.
        Parameters:
        -----------
        X : numpy.ndarray
            The test data.
        y : numpy.ndarray
            The target values.

        Returns:
        --------
        score : float
            The score of the model.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
