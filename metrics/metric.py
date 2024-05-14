import numpy as np

class Metric:
    def __init__(self, name: str, value: float):
        pass 
    
    
    def score(self, y_true, y_pred):
        raise NotImplementedError("score method not implemented")
    
    
    
class Accuracy(Metric):
    def __init__(self):
        pass
    
    def score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)


class Recall(Metric):
    def __init__(self):
        pass
    
    def score(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn)
    
class Precision(Metric):
    def __init__(self):
        pass
    
    def score(self, y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp)

class F1Score(Metric):
    def __init__(self):
        pass
    
    def score(self, y_true, y_pred):
        precision = Precision().score(y_true, y_pred)
        recall = Recall().score(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall)
    

class MeanSquaredError(Metric):
    def __init__(self):
        pass
    
    def score(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
class MeanAbsoluteError(Metric):
    def __init__(self):
        pass
    
    def score(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    

