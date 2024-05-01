

class Estimator: 
    """This is the base class for all estimators in the package. It provides the basic structure for all estimators.
    """
    
    def __init__(self):
        """This is the constructor of the class.
        """
        pass
    
    def fit(self, X, y=None):
        """This method is used to train the model on the training data.
        Parameters:
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The target values.
        Returns:
        self: The trained model.
        """
        raise NotImplementedError("The fit method is not implemented.")
    
    def get_params(self):
        """This method is used to get the parameters of the model.
        Returns:
        dict: The parameters of the model.
        """
        raise NotImplementedError("The get_params method is not implemented.")
    
    def set_params(self, **params):
        """This method is used to set the parameters of the model.
        Parameters:
        **params: The parameters of the model.
        """
        raise NotImplementedError("The set_params method is not implemented.")
    
        