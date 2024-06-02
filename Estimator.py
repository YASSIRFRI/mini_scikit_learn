class Estimator:
    """
    Base class for all estimators in the package. It provides the basic structure for all estimators.

    Methods:
    --------
    fit(X, y=None)
        Train the model on the training data.
    get_params()
        Get the parameters of the model.
    set_params(**params)
        Set the parameters of the model.
    """
    
    def __init__(self):
        """
        Initialize the estimator.
        """
        pass
    
    def fit(self, X, y=None):
        """
        Train the model on the training data.
        Parameters:
        -----------
        X : numpy.ndarray
            The training data.
        y : numpy.ndarray, optional
            The target values. Default is None.

        Returns:
        --------
        self : object
            The trained model.

        Raises:
        -------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("The fit method is not implemented.")
    
    def get_params(self):
        """
        Get the parameters of the model.
        Returns:
        --------
        params : dict
            The parameters of the model.

        Raises:
        -------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("The get_params method is not implemented.")
    
    def set_params(self, **params):
        """
        Set the parameters of the model.
        Parameters:
        -----------
        **params : dict
            The parameters of the model.
        Raises:
        -------
        NotImplementedError
            If the method is not implemented.
        """
        raise NotImplementedError("The set_params method is not implemented.")
