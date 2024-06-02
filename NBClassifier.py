import numpy as np

class NBClassifier:
    """
    Gaussian Naive Bayes Classifier.

    Attributes:
    -----------
    classes : numpy.ndarray or None
        Unique class labels.
    class_statistics : dict
        Dictionary to store mean and standard deviation of features for each class.
    class_priors : dict
        Dictionary to store the prior probabilities for each class.
    """

    def __init__(self):
        """Initialize the Gaussian Naive Bayes Classifier."""
        self.classes = None
        self.class_statistics = {}
        self.class_priors = {}

    def fit(self, X, y):
        """
        Fit the Gaussian Naive Bayes model according to the given training data.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data.
        y : numpy.ndarray
            Target labels.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            cls_indices = (y == cls)
            cls_X = X[cls_indices]
            self.class_statistics[cls] = {
                'mean': np.mean(cls_X, axis=0),
                'std': np.std(cls_X, axis=0) + 1e-3  # Add a small value to avoid division by zero
            }
            self.class_priors[cls] = np.mean(cls_indices)
        return self

    def _calculate_likelihood(self, x, mean, std):
        """
        Calculate the likelihood of feature values given class parameters.

        Parameters:
        -----------
        x : numpy.ndarray
            Feature values.
        mean : numpy.ndarray
            Mean of the feature values for a class.
        std : numpy.ndarray
            Standard deviation of the feature values for a class.

        Returns:
        --------
        numpy.ndarray
            Likelihood of the feature values.
        """
        exponent = -0.5 * ((x - mean) / std) ** 2
        return np.exp(exponent) / (np.sqrt(2 * np.pi) * std)

    def _calculate_class_posteriors(self, x):
        """
        Calculate posteriors for all classes given feature values.

        Parameters:
        -----------
        x : numpy.ndarray
            Feature values.

        Returns:
        --------
        posteriors : dict
            Dictionary of posterior probabilities for each class.
        """
        posteriors = {}
        for cls in self.classes:
            mean, std = self.class_statistics[cls]['mean'], self.class_statistics[cls]['std']
            likelihood = self._calculate_likelihood(x, mean, std)
            prior = self.class_priors[cls]
            posteriors[cls] = prior * np.prod(likelihood) 
        return posteriors

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.

        Returns:
        --------
        numpy.ndarray
            Predicted class labels.
        """
        if self.classes is None:
            raise ValueError("Classifier not fitted yet.")
        predictions = [max(self._calculate_class_posteriors(x), key=self._calculate_class_posteriors(x).get) for x in X]
        return np.array(predictions)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : numpy.ndarray
            Test data.
        y : numpy.ndarray
            True labels for X.

        Returns:
        --------
        float
            Mean accuracy of self.predict(X) wrt. y.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
