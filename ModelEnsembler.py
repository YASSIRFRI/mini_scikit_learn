import numpy as np
from . import Estimator
from . import Predictor


class ModelEnsembler(Predictor.Predictor, Estimator.Estimator):
    def __init__(self, models):
        """This is the constructor of the class.
        Parameters:
        models (list): A list of models to ensemble.
        """
        self.models = models
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.mean([model.predict(X) for model in self.models], axis=0)

    def score(self, X, y):
        """This method is used to evaluate the model on the test data."""
        y_pred = self.predict(X)
        return np.mean(y == y_pred)


class VotingEnsembler(Predictor.Predictor, Estimator.Estimator):
    def __init__(self, models):
        """This is the constructor of the class.
        Parameters:
        models (list): A list of models to ensemble.
        """
        self.models = models
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
            print(model.predict(X).shape)
            print(model.score(X, y))
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        predictions = np.array([model.predict(X) for model in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)


    def score(self, X, y):
        """This method is used to evaluate the model on the test data."""
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    


class BaggingEnsembler(Predictor.Predictor, Estimator.Estimator):
    def __init__(self, model, n_estimators=10):
        """This is the constructor of the class.
        Parameters:
        model (Estimator): The base model to ensemble.
        n_estimators (int): The number of base models to train.
        """
        self.model = model
        self.n_estimators = n_estimators
        self.models = [model.__class__() for _ in range(n_estimators)]
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            X_resampled, y_resampled = resample(X, y)
            model.fit(X_resampled, y_resampled)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.mean([model.predict(X) for model in self.models], axis=0)

    def score(self, X, y):
        """This method is used to evaluate the model on the test data."""
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    


class StackingEnsembler(Predictor.Predictor, Estimator.Estimator):
    def __init__(self, base_models, meta_model):
        """This is the constructor of the class.
        Parameters:
        base_models (list): A list of base models to ensemble.
        meta_model (Estimator): The meta model that combines the base models' predictions.
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y, n_iterations=1000):
        """
        
        This method is used to train the model on the training data.
        Parameters:
        X (numpy.ndarray): The training data.
        y (numpy.ndarray): The target values.
        Returns:
        self: The trained model.
        
        """
        for model in self.base_models:
            model.fit(X, y)
            print("Accuracy of Model: ",base_models.index(model))
            print(model.score(X, y))
        X_meta = np.array([model.predict(X) for model in self.base_models]).T
        self.meta_model.fit(X_meta, y)
        print("Meta model score")
        print(self.meta_model.score(X_meta, y))
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        This method is used to make predictions on the test data.
        Parameters:
        X (numpy.ndarray): The test data.
        Returns:
        numpy.ndarray: The predictions.
        
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        X_meta = np.array([model.predict(X) for model in self.base_models]).T
        answer=self.meta_model.predict(X_meta)
        return answer

    def score(self, X, y):
        """This method is used to evaluate the model on the test data."""
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    
    