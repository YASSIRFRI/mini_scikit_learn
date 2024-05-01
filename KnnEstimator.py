import numpy as np
import Estimator
import Predictor


class KnnEstimator(Predictor.Predictor, Estimator.Estimator):
    
    def __init__(self, n_neighbors=5):
        """This is the constructor of the class.
        Parameters:
        n_neighbors (int): The number of neighbors to consider for prediction.
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.is_fitted = False
        
    def fit(self, X, y):
        """This method is used to train the model on the training data.
        Parameters:
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The target values.
        Returns:
        self: The trained model.
        """
        self.X_train = X
        self.y_train = y
        self.is_fitted = True
        return self
    
    def get_params(self):
        """This method is used to get the parameters of the model.
        Returns:
        dict: The parameters of the model.
        """
        return {"n_neighbors": self.n_neighbors}
    
    def predict(self, X):
        """This method is used to make predictions on the test data.
        Parameters:
        X (numpy.ndarray): The test data.
        Returns:
        numpy.ndarray: The predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            distances = np.linalg.norm(self.X_train - X[i], axis=1)
            nearest_neighbors = np.argsort(distances)[:self.n_neighbors]
            y_pred[i] = np.mean(self.y_train[nearest_neighbors])
        return y_pred