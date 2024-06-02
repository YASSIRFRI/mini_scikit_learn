from __future__ import print_function, division
import numpy as np
import math

class Sigmoid:
    """
    Sigmoid activation function.

    Methods:
    --------
    __call__(x)
        Compute the sigmoid function for the input x.
    gradient(x)
        Compute the gradient of the sigmoid function for the input x.
    """

    def __call__(self, x):
        """
        Compute the sigmoid function.

        Parameters:
        -----------
        x : numpy.ndarray
            The input data.

        Returns:
        --------
        numpy.ndarray
            The sigmoid of the input data.
        """
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        """
        Compute the gradient of the sigmoid function.

        Parameters:
        -----------
        x : numpy.ndarray
            The input data.

        Returns:
        --------
        numpy.ndarray
            The gradient of the sigmoid function.
        """
        return self.__call__(x) * (1 - self.__call__(x))

class LogisticRegression:
    """
    Logistic Regression classifier.

    Parameters:
    -----------
    learning_rate : float, optional (default=0.1)
        The step length that will be taken when following the negative gradient during training.
    gradient_descent : bool, optional (default=True)
        True if gradient descent should be used when training. If False, batch optimization by least squares is used.

    Attributes:
    -----------
    param : numpy.ndarray or None
        The parameters (weights) of the logistic regression model.
    learning_rate : float
        The step length that will be taken when following the negative gradient during training.
    gradient_descent : bool
        Indicates if gradient descent should be used when training.
    sigmoid : Sigmoid
        An instance of the Sigmoid class.
    """

    def __init__(self, learning_rate=0.1, gradient_descent=True):
        """Initialize the logistic regression model."""
        self.param = None
        self.learning_rate = learning_rate
        self.gradient_descent = gradient_descent
        self.sigmoid = Sigmoid()

    def _initialize_parameters(self, X):
        """
        Initialize the parameters (weights) of the logistic regression model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data.

        Attributes:
        -----------
        param : numpy.ndarray
            The initialized parameters (weights) of the logistic regression model.
        """
        n_features = np.shape(X)[1]
        limit = 1 / math.sqrt(n_features)
        self.param = np.random.uniform(-limit, limit, (n_features,))

    def fit(self, X, y, n_iterations=4000):
        """
        Train the logistic regression model.

        Parameters:
        -----------
        X : numpy.ndarray
            The training data.
        y : numpy.ndarray
            The target values.
        n_iterations : int, optional (default=4000)
            The number of iterations to run the training loop.
        """
        self._initialize_parameters(X)
        for i in range(n_iterations):
            y_pred = self.sigmoid(X.dot(self.param))
            if self.gradient_descent:
                self.param -= self.learning_rate * -(y - y_pred).dot(X)
            else:
                diag_gradient = self.make_diagonal(self.sigmoid.gradient(X.dot(self.param)))
                self.param = np.linalg.pinv(X.T.dot(diag_gradient).dot(X)).dot(X.T).dot(diag_gradient.dot(X).dot(self.param) + y - y_pred)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data.

        Returns:
        --------
        numpy.ndarray
            The predicted class labels.
        """
        y_pred = np.round(self.sigmoid(X.dot(self.param))).astype(int)
        return y_pred

    def make_diagonal(self, x):
        """
        Convert a vector into a diagonal matrix.

        Parameters:
        -----------
        x : numpy.ndarray
            The input vector.

        Returns:
        --------
        numpy.ndarray
            The diagonal matrix.
        """
        m = np.zeros((len(x), len(x)))
        for i in range(len(m[0])):
            m[i, i] = x[i]
        return m

    def score(self, X, y):
        """
        Compute the accuracy of the model.

        Parameters:
        -----------
        X : numpy.ndarray
            The test data.
        y : numpy.ndarray
            The true labels.

        Returns:
        --------
        float
            The accuracy of the model.
        """
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return accuracy
