import Estimator
import numpy as np
import Predictor


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

    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
            print(model.score(X, y))
        X_meta = np.array([model.predict(X) for model in self.base_models]).T
        self.meta_model.fit(X_meta, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        X_meta = np.array([model.predict(X) for model in self.base_models]).T
        return self.meta_model.predict(X_meta)

    def score(self, X, y):
        """This method is used to evaluate the model on the test data."""
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
    
    