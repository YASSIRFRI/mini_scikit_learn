import numpy as np
import Predictor
import Estimator




class LinearModel(Predictor.Predictor,Estimator.Estimator):
    
    def __init__(self, fit_intercept=True):
        """This is the constructor of the class.
        Parameters:
        fit_intercept (bool): Whether to fit an intercept term in the model.
        """
        self.fit_intercept = fit_intercept
        self.beta = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """This method is used to train the model on the training data.
        Parameters:
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The target values.
        Returns:
        self: The trained model.
        """
        self.is_fitted = True
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return self
    
    
    def get_params(self):
        """This method is used to get the parameters of the model.
        Returns:
        dict: The parameters of the model.
        """
        return {"fit_intercept": self.fit_intercept}
        
    
    def predict(self, X):
        """This method is used to make predictions on the test data.
        Parameters:
        X (numpy.ndarray): The test data.
        Returns:
        numpy.ndarray: The predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    



class LinearRegression(LinearModel):
    def __init__(self, fit_intercept=True):
        """This is the constructor of the class.
        Parameters:
        fit_intercept (bool): Whether to fit an intercept term in the model.
        """
        super().__init__(self)
    
    def score(self, X, y):
        """This method is used to evaluate the model on the test data.
        Parameters:
        X (numpy.ndarray): The test data.
        y (numpy.ndarray): The target values.
        Returns:
        float: The score of the model.
        """
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    def get_params(self):
        """This method is used to get the parameters of the model.
        Returns:
        dict: The parameters of the model.
        """
        return self.model.get_params()
    
    def set_params(self, **params):
        """This method is used to set the parameters of the model.
        Parameters:
        **params: The parameters of the model.
        """
        self.model.fit_intercept = params["fit_intercept"]
        return self
    
    


class LogisticRegression:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """This method is used to train the model on the training data.
        Parameters:
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The target values (binary 0/1).
        Returns:
        self: The trained model.
        """
        
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict_proba(self, X):
        """This method is used to predict the probabilities of class 1.
        Parameters:
        X (numpy.ndarray): The test data.
        Returns:
        numpy.ndarray: The predicted probabilities.
        """
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return self._sigmoid(X @ self.beta)

    def predict(self, X, threshold=0.5):
        """This method is used to predict the class labels.
        Parameters:
        X (numpy.ndarray): The test data.
        threshold (float): Threshold for classification (default: 0.5).
        Returns:
        numpy.ndarray: The predicted class labels (binary 0/1).
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)    