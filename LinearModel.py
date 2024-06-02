import numpy as np
from . import Estimator
from . import Predictor
from numpy import log, dot, exp, shape
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LinearModel(Predictor.Predictor, Estimator.Estimator):
    """
    A linear model for regression or classification.

    Parameters:
    -----------
    fit_intercept : bool, optional (default=True)
        Whether to fit an intercept term in the model.

    Attributes:
    -----------
    fit_intercept : bool
        Whether to fit an intercept term in the model.
    beta : numpy.ndarray or None
        The coefficients of the linear model.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """
    
    def __init__(self, fit_intercept=True):
        """Initialize the linear model."""
        self.fit_intercept = fit_intercept
        self.beta = None
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
        self.is_fitted = True
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y
        return self
    
    def get_params(self):
        """
        Get the parameters of the model.

        Returns:
        --------
        params : dict
            The parameters of the model.
        """
        return {"fit_intercept": self.fit_intercept}
    
    def predict(self, X):
        """
        Make predictions on the test data.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta

class LinearRegression(LinearModel):
    """
    A linear regression model.

    Parameters:
    -----------
    fit_intercept : bool, optional (default=True)
        Whether to fit an intercept term in the model.
    """
    
    def __init__(self, fit_intercept=True):
        """Initialize the linear regression model."""
        super().__init__(fit_intercept)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    def get_params(self):
        return super().get_params()
    
    def set_params(self, **params):
        """This method is used to set the parameters of the model.
        Parameters:
        **params: The parameters of the model.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self

class LogisticRegression:
    """
    Logistic Regression Classifier.

    Parameters:
    -----------
    max_iter : int, optional (default=10000)
        Maximum number of iterations for the optimization loop.
    thres : float, optional (default=1e-3)
        Threshold for the optimization convergence.

    Attributes:
    -----------
    max_iter : int
        Maximum number of iterations for the optimization loop.
    thres : float
        Threshold for the optimization convergence.
    weights : numpy.ndarray or None
        The weights of the logistic regression model.
    classes : numpy.ndarray
        The unique class labels.
    class_labels : dict
        Dictionary mapping class labels to indices.
    loss : list
        List to store the loss values during training.
    """
    
    def __init__(self, max_iter=10000, thres=1e-3):
        self.max_iter = max_iter
        self.thres = thres
    
    def fit(self, X, y, batch_size=64, lr=0.001, rand_seed=4, verbose=False): 
        """
        Fit the logistic regression model to the training set.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Training labels.
        batch_size : int, optional (default=64)
            Batch size for training.
        lr : float, optional (default=0.001)
            Learning rate for gradient descent.
        rand_seed : int, optional (default=4)
            Random seed for reproducibility.
        verbose : bool, optional (default=False)
            Whether to print training progress.

        Returns:
        --------
        self : object
            Fitted logistic regression model.
        """
        np.random.seed(rand_seed) 
        self.classes = np.unique(y)
        self.class_labels = {c: i for i, c in enumerate(self.classes)}
        X = self.add_bias(X)
        y = self.one_hot(y)
        self.loss = []
        self.weights = np.zeros(shape=(len(self.classes), X.shape[1]))
        self.fit_data(X, y, batch_size, lr, verbose)
        return self

    def fit_data(self, X, y, batch_size, lr, verbose):
        """
        Train the logistic regression model using batch gradient descent.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Training labels.
        batch_size : int
            Batch size for training.
        lr : float
            Learning rate for gradient descent.
        verbose : bool
            Whether to print training progress.
        """
        i = 0
        while (not self.max_iter or i < self.max_iter):
            self.loss.append(self.cross_entropy(y, self.predict_(X)))
            idx = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[idx], y[idx]
            error = y_batch - self.predict_(X_batch)
            update = (lr * np.dot(error.T, X_batch))
            self.weights += update
            if np.abs(update).max() < self.thres:
                break
            if i % 1000 == 0 and verbose: 
                print(f'Training Accuracy at {i} iterations is {self.evaluate_(X, y)}')
            i += 1
    
    def predict_probs(self, X):
        """
        Predict probabilities for given data using the logistic regression model.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict.

        Returns:
        --------
        probs : numpy.ndarray
            Predicted probabilities.
        """
        return self.predict_(self.add_bias(X))
    
    def predict_(self, X):
        """Predict probabilities for given data.
        """
        pre_vals = np.dot(X, self.weights.T).reshape(-1, len(self.classes))
        return self.softmax(pre_vals)
    
    def softmax(self, z):
        """
        Compute the softmax of a set of values..
        """
        return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)

    def predict(self, X):
        """
        Predict class labels for given data.
        """
        self.probs_ = self.predict_probs(X)
        return np.vectorize(lambda c: self.classes[c])(np.argmax(self.probs_, axis=1))
  
    def add_bias(self, X):
        """
        Add bias term to the data.

        Parameters:
        -----------
        X : numpy.ndarray
            Data to add bias term.

        Returns:
        --------
        X_with_bias : numpy.ndarray
            Data with bias term added.
        """
        return np.insert(X, 0, 1, axis=1)
  
    def get_random_weights(self, row, col):
        """
        Get random weights for initialization.

        Parameters:
        -----------
        row : int
            Number of rows.
        col : int
            Number of columns.

        Returns:
        --------
        weights : numpy.ndarray
            Random weights.
        """
        return np.zeros(shape=(row, col))

    def one_hot(self, y):
        """
        Convert class labels to one-hot encoding.

        Parameters:
        -----------
        y : numpy.ndarray
            Class labels.

        Returns:
        --------
        y_one_hot : numpy.ndarray
            One-hot encoded labels.
        """
        return np.eye(len(self.classes))[np.vectorize(lambda c: self.class_labels[c])(y).reshape(-1)]
    
    def score(self, X, y):
        """
        Compute the accuracy of the model.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.
        y : numpy.ndarray
            True labels.

        Returns:
        --------
        accuracy : float
            Accuracy of the model.
        """
        return np.mean(self.predict(X) == y)
    
    def evaluate_(self, X, y):
        """
        Evaluate the model during training.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Training labels.

        Returns:
        --------
        accuracy : float
            Accuracy of the model.
        """
        return np.mean(np.argmax(self.predict_(X), axis=1) == np.argmax(y, axis=1))
    
    def cross_entropy(self, y, probs):
        """
        Compute the cross-entropy loss.

        Parameters:
        -----------
        y : numpy.ndarray
            True labels.
        probs : numpy.ndarray
            Predicted probabilities.

        Returns:
        --------
        loss : float
            Cross-entropy loss.
        """
        return -1 * np.mean(y * np.log(probs))

    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)