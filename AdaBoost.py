import numpy as np
import math
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

class DecisionStump():
    """
    Decision Stump classifier for decision trees with a maximum depth of one.

    Attributes:
    -----------
    polarity : int
        Indicates the direction of the inequality to compare feature values.
    feature_index : int
        The index of the feature used to make the split.
    threshold : float
        The threshold value used to split the feature.
    alpha : float
        The weight assigned to this classifier in the final prediction.
    """
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

class Adaboost():
    """
    Boosting method that uses a number of weak classifiers in ensemble to make a strong classifier.
    This implementation uses decision stumps, which are one-level decision trees.

    Parameters:
    -----------
    n_clf : int
        The number of weak classifiers that will be used.

    Attributes:
    -----------
    n_clf : int
        The number of weak classifiers that will be used.
    clfs : list
        List to store the weak classifiers (decision stumps) used in the ensemble.
    """
    def __init__(self, n_clf=5):
        self.n_clf = n_clf

    def fit(self, X, y):
        """
        Fit the model using the given training data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).
        y : numpy.ndarray
            The target values of shape (n_samples,).
        """
        n_samples, n_features = np.shape(X)
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                for threshold in unique_values:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[X[:, feature_i] < threshold] = -1
                    error = sum(w[y != prediction])
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = feature_i
                        min_error = error
            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            predictions = np.ones(np.shape(y))
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)
            self.clfs.append(clf)

    def predict(self, X):
        """
        Predict the class labels for the input data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input features of shape (n_samples, n_features).

        Returns:
        --------
        y_pred : numpy.ndarray
            The predicted class labels of shape (n_samples,).
        """
        n_samples = np.shape(X)[0]
        y_pred = np.zeros((n_samples, 1))
        for clf in self.clfs:
            # Set all predictions to '1' initially
            predictions = np.ones(np.shape(y_pred))
            # The indexes where the sample values are below threshold
            negative_idx = (clf.polarity * X[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions

        y_pred = np.sign(y_pred).flatten()

        return y_pred