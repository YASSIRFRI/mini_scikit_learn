from .Estimator import Estimator
class Predictor(Estimator):
    
    def __init__(self, model=None):
        """Initialize the Predictor with a given model."""
        self.model = model
        
    def predict(self, X):
        """
        Make predictions on the test data.
        Parameters:
        -----------
        X : numpy.ndarray
            The test data.
        Returns:
        --------
        numpy.ndarray
            The predictions.
        """
        return self.model.predict(X)
    
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
        float
            The score of the model.
        Raises:
        -------
        NotImplementedError
            If the score method has not been implemented.
        """
        raise NotImplementedError("The score method has not been implemented.")
    
    def fit_predict(self, X_train, y_train, X_test):
        """
        Train the model on the training data and make predictions on the test data.
        Parameters:
        -----------
        X_train : numpy.ndarray
            The training data.
        y_train : numpy.ndarray
            The target values.
        X_test : numpy.ndarray
            The test data.

        Returns:
        --------
        numpy.ndarray
            The predictions.
        """
        self.model.fit(X_train, y_train)
        return self.predict(X_test)
    
    
    
    
    
    
