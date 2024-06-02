import Estimator
import numpy as np
import Predictor
from sklearn.utils import resample

class ModelEnsembler(Predictor.Predictor, Estimator.Estimator):
    """
    Model Ensembler for averaging predictions.

    Parameters:
    -----------
    models : list
        A list of models to ensemble.

    Attributes:
    -----------
    models : list
        A list of models to ensemble.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """

    def __init__(self, models):
        """Initialize the Model Ensembler."""
        self.models = models
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        """
        Fit the ensembled models.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns:
        --------
        self : object
            Fitted ensembled model.
        """
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the ensembled models.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.

        Returns:
        --------
        numpy.ndarray
            Averaged predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.mean([model.predict(X) for model in self.models], axis=0)

    def score(self, X, y):
        """
        Evaluate the ensembled model.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.
        y : numpy.ndarray
            True labels.

        Returns:
        --------
        float
            Accuracy of the ensembled model.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

class VotingEnsembler(Predictor.Predictor, Estimator.Estimator):
    """
    Voting Ensembler for majority vote predictions.

    Parameters:
    -----------
    models : list
        A list of models to ensemble.

    Attributes:
    -----------
    models : list
        A list of models to ensemble.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """

    def __init__(self, models):
        """Initialize the Voting Ensembler."""
        self.models = models
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        """
        Fit the ensembled models.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns:
        --------
        self : object
            Fitted ensembled model.
        """
        for model in self.models:
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the ensembled models.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.

        Returns:
        --------
        numpy.ndarray
            Majority vote predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        predictions = np.array([model.predict(X) for model in self.models])
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions)

    def score(self, X, y):
        """
        Evaluate the ensembled model.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.
        y : numpy.ndarray
            True labels.

        Returns:
        --------
        float
            Accuracy of the ensembled model.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

class BaggingEnsembler(Predictor.Predictor, Estimator.Estimator):
    """
    Bagging Ensembler for averaging predictions with bootstrap sampling.

    Parameters:
    -----------
    model : Estimator
        The base model to ensemble.
    n_estimators : int, optional (default=10)
        The number of base models to train.

    Attributes:
    -----------
    model : Estimator
        The base model to ensemble.
    n_estimators : int
        The number of base models to train.
    models : list
        A list of base models.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """

    def __init__(self, model, n_estimators=10):
        """Initialize the Bagging Ensembler."""
        self.model = model
        self.n_estimators = n_estimators
        self.models = [model.__class__() for _ in range(n_estimators)]
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        """
        Fit the ensembled models using bootstrap samples.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns:
        --------
        self : object
            Fitted ensembled model.
        """
        for i, model in enumerate(self.models):
            X_resampled, y_resampled = resample(X, y)
            model.fit(X_resampled, y_resampled)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the ensembled models.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.

        Returns:
        --------
        numpy.ndarray
            Averaged predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        return np.mean([model.predict(X) for model in self.models], axis=0)

    def score(self, X, y):
        """
        Evaluate the ensembled model.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.
        y : numpy.ndarray
            True labels.

        Returns:
        --------
        float
            Accuracy of the ensembled model.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)

class StackingEnsembler(Predictor.Predictor, Estimator.Estimator):
    """
    Stacking Ensembler for combining base model predictions with a meta model.

    Parameters:
    -----------
    base_models : list
        A list of base models to ensemble.
    meta_model : Estimator
        The meta model that combines the base models' predictions.

    Attributes:
    -----------
    base_models : list
        A list of base models to ensemble.
    meta_model : Estimator
        The meta model that combines the base models' predictions.
    is_fitted : bool
        Indicates whether the model has been fitted.
    """

    def __init__(self, base_models, meta_model):
        """Initialize the Stacking Ensembler."""
        self.base_models = base_models
        self.meta_model = meta_model
        self.is_fitted = False
        super().__init__(self)

    def fit(self, X, y):
        """
        Fit the base models and the meta model.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target values.

        Returns:
        --------
        self : object
            Fitted ensembled model.
        """
        for model in self.base_models:
            model.fit(X, y)
        X_meta = np.array([model.predict(X) for model in self.base_models]).T
        self.meta_model.fit(X_meta, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Make predictions using the ensembled models and the meta model.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.

        Returns:
        --------
        numpy.ndarray
            Meta model predictions.
        """
        if not self.is_fitted:
            raise ValueError("The model has not been fitted yet.")
        X_meta = np.array([model.predict(X) for model in self.base_models]).T
        return self.meta_model.predict(X_meta)

    def score(self, X, y):
        """
        Evaluate the ensembled model.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.
        y : numpy.ndarray
            True labels.

        Returns:
        --------
        float
            Accuracy of the ensembled model.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
