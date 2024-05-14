import numpy as np
import Predictor
import Estimator
from numpy import log, dot, exp, shape
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split




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
    
    






import numpy as np

class LogisticRegression:
    
    def __init__(self):
        self.w = None
        self.b = None

    def sigmoid(self, z):
        """
        Implement the sigmoid function.

        Arguments:
        z -- A scalar or numpy array of any size.

        Returns:
        s -- Sigmoid of z.
        """
        s = 1.0 / (1.0 + np.exp(-z))
        return s

    def initialize(self, dim):
        """
        Initialize the weights and the bias.

        Arguments:
        dim -- Size of the weight vector.

        Returns:
        w -- Initialized weights (numpy array of shape (dim, 1)).
        b -- Initialized bias (scalar).
        """
        w = np.zeros((dim, 1))
        b = 0.0
        return w, b

    def propagate(self, w, b, X, Y):
        """
        Implement the cost function and its gradient.

        Arguments:
        w -- Weights (numpy array of shape (dim, 1)).
        b -- Bias (scalar).
        X -- Data (numpy array of shape (dim, number of examples)).
        Y -- True labels (numpy array of shape (1, number of examples)).

        Returns:
        grads -- Dictionary containing the gradients of the weights and bias.
        cost -- Cost function value.
        """
        m = X.shape[1]
        z = np.dot(w.T, X) + b
        A = self.sigmoid(z)
        cost = -1.0 / m * np.sum(Y * np.log(A) + (1.0 - Y) * np.log(1.0 - A))
        dw = 1.0 / m * np.dot(X, (A - Y).T)
        db = 1.0 / m * np.sum(A - Y)
        grads = {"dw": dw, "db": db}
        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        """
        Optimize weights and bias using gradient descent.

        Arguments:
        w -- Weights (numpy array of shape (dim, 1)).
        b -- Bias (scalar).
        X -- Data (numpy array of shape (dim, number of examples)).
        Y -- True labels (numpy array of shape (1, number of examples)).
        num_iterations -- Number of iterations for the optimization loop.
        learning_rate -- Learning rate for gradient descent.
        print_cost -- If True, print the cost every 100 iterations.

        Returns:
        params -- Dictionary containing the optimized weights and bias.
        grads -- Dictionary containing the gradients of the weights and bias.
        costs -- List of costs at every 100 iterations.
        """
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - learning_rate * dw
            b = b - learning_rate * db
            if i % 100 == 0:
                costs.append(cost)
                if print_cost:
                    print(f"Cost after iteration {i}: {cost}")
        params = {"w": w, "b": b}
        return params, grads, costs

    def fit(self, X_train, Y_train, num_iterations=1000, learning_rate=0.5, print_cost=False):
        """
        Fit the logistic regression model to the training set.

        Arguments:
        X_train -- Training data (numpy array of shape (dim, number of examples)).
        Y_train -- Training labels (numpy array of shape (1, number of examples)).
        num_iterations -- Number of iterations for the optimization loop.
        learning_rate -- Learning rate for gradient descent.
        print_cost -- If True, print the cost every 100 iterations.

        Returns:
        self -- Fitted logistic regression model.
        """
        dim = X_train.shape[0]
        self.w, self.b = self.initialize(dim)
        parameters, grads, costs = self.optimize(self.w, self.b, X_train, Y_train, num_iterations, learning_rate, print_cost)
        self.w = parameters["w"]
        self.b = parameters["b"]
        return self

    def predict(self, X):
        """
        Predict labels for given data using the logistic regression model.

        Arguments:
        X -- Data to predict (numpy array of shape (dim, number of examples)).

        Returns:
        Y_prediction -- Predicted labels (numpy array of shape (1, number of examples)).
        """
        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)
        Y_prediction = (A > 0.5).astype(int)
        return Y_prediction

    def evaluate(self, X_train, Y_train, X_test, Y_test):
        """
        Evaluate the logistic regression model on training and test sets.

        Arguments:
        X_train -- Training data (numpy array of shape (dim, number of examples)).
        Y_train -- Training labels (numpy array of shape (1, number of examples)).
        X_test -- Test data (numpy array of shape (dim, number of examples)).
        Y_test -- Test labels (numpy array of shape (1, number of examples)).

        Returns:
        train_accuracy -- Accuracy on the training set.
        test_accuracy -- Accuracy on the test set.
        """
        Y_prediction_train = self.predict(X_train)
        Y_prediction_test = self.predict(X_test)
        train_accuracy = 100.0 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100.0
        test_accuracy = 100.0 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100.0
        print("Training accuracy:", train_accuracy)
        print("Test accuracy:", test_accuracy)
        return train_accuracy, test_accuracy
