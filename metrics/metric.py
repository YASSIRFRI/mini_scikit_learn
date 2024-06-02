import numpy as np

class Metric:
    """
    Base class for metrics.
    
    Attributes:
        name (str): The name of the metric.
        value (float): The value of the metric.
    """
    def __init__(self, name: str, value: float):
        """
        Initializes the Metric class.

        Parameters:
            name (str): The name of the metric.
            value (float): The value of the metric.
        """
        pass 
    
    def score(self, y_true, y_pred):
        """
        Calculates the score for the metric.
        
        Parameters:
            y_true (array-like): True labels or values.
            y_pred (array-like): Predicted labels or values.
        
        Returns:
            float: The calculated score.
        
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("score method not implemented")
    

class Accuracy(Metric):
    """
    Accuracy metric class.
    """
    def __init__(self):
        """
        Initializes the Accuracy class.
        """
        pass
    
    def score(self, y_true, y_pred):
        """
        Calculates the accuracy score.
        
        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
        
        Returns:
            float: The accuracy score.
        """
        return np.mean(y_true == y_pred)


class Recall(Metric):
    """
    Recall metric class.
    """
    def __init__(self):
        """
        Initializes the Recall class.
        """
        pass
    
    def score(self, y_true, y_pred):
        """
        Calculates the recall score.
        
        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
        
        Returns:
            float: The recall score.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn)
    

class Precision(Metric):
    """
    Precision metric class.
    """
    def __init__(self):
        """
        Initializes the Precision class.
        """
        pass
    
    def score(self, y_true, y_pred):
        """
        Calculates the precision score.
        
        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
        
        Returns:
            float: The precision score.
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp)


class F1Score(Metric):
    """
    F1 Score metric class.
    """
    def __init__(self):
        """
        Initializes the F1Score class.
        """
        pass
    
    def score(self, y_true, y_pred):
        """
        Calculates the F1 score.
        
        Parameters:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
        
        Returns:
            float: The F1 score.
        """
        precision = Precision().score(y_true, y_pred)
        recall = Recall().score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)
    

class MeanSquaredError(Metric):
    """
    Mean Squared Error metric class.
    """
    def __init__(self):
        """
        Initializes the MeanSquaredError class.
        """
        pass
    
    def score(self, y_true, y_pred):
        """
        Calculates the mean squared error score.
        
        Parameters:
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.
        
        Returns:
            float: The mean squared error score.
        """
        return np.mean((y_true - y_pred) ** 2)
    

class MeanAbsoluteError(Metric):
    """
    Mean Absolute Error metric class.
    """
    def __init__(self):
        """
        Initializes the MeanAbsoluteError class.
        """
        pass
    
    def score(self, y_true, y_pred):
        """
        Calculates the mean absolute error score.
        
        Parameters:
            y_true (array-like): True values.
            y_pred (array-like): Predicted values.
        
        Returns:
            float: The mean absolute error score.
        """
        return np.mean(np.abs(y_true - y_pred))
