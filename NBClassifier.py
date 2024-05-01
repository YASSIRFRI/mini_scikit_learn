import Estimator
import numpy as np
import Predictor



class NBClassifier(Predictor.Predictor, Estimator.Estimator):
    
    def __init__(self):
        """This is the constructor of the class.
        """
        self.class_priors = None
        self.class_conditional
        
    def fit(self, X, y):
        """This method is used to train the model on the training data.
        Parameters:
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The target values.
        Returns:
        self: The trained model.
        """
        self.class_priors = np.bincount(y) / len(y)
        self.class_conditional = []
        for i in range(len(np.unique(y))):
            X_class = X[y == i]
            self.class_conditional
            for j in range(X.shape[1]):
                self.class_conditional
                self.class_conditional
        return self
    
    def get_params(self):
        """This method is used to get the parameters of the model.
        Returns:
        dict: The parameters of the model.
        """
        return {}
    
    def predict(self, X):
        """This method is used to make predictions on the test data.
        Parameters:
        X (numpy.ndarray): The test data.
        Returns:
        numpy.ndarray: The predictions.
        """
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            posteriors = []
            for j in range(len(np.unique(y))):
                prior = self.class_priors[j]
                likelihood = 1
                for k in range(X.shape[1]):
                    likelihood *= self.class_conditional[j][k][X[i, k]]
                posterior = prior * likelihood
                posteriors.append(posterior)
            y_pred[i] = np.argmax(posteriors)
        return y_pred
    
    
        