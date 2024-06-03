import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoostClassifier:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        """
        Initializes the AdaBoost classifier.

        Parameters:
        - n_estimators: int, number of weak classifiers to use.
        - learning_rate: float, learning rate to scale the contribution of each classifier.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.alphas = []

    def fit(self, X, y):
        """
        Trains the AdaBoost classifier.

        Parameters:
        - X: array-like, shape (n_samples, n_features), training data.
        - y: array-like, shape (n_samples,), target values.
        """
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=1)
            model.fit(X, y, sample_weight=w)
            y_pred = model.predict(X)
            incorrect = (y_pred != y)
            err = np.dot(w, incorrect) / np.sum(w)

            if err == 0:
                alpha = np.inf
            else:
                alpha = self.learning_rate * np.log((1 - err) / err)

            w *= np.exp(alpha * incorrect)
            w /= np.sum(w)

            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X):
        """
        Predicts the class labels for the given data.

        Parameters:
        - X: array-like, shape (n_samples, n_features), input data.

        Returns:
        - y_pred: array, shape (n_samples,), predicted class labels.
        """
        model_preds = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.dot(self.alphas, model_preds)
        y_pred = np.sign(weighted_preds)
        return y_pred

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    digits = load_digits()
    X, y = digits.data, digits.target
    y = np.where(y % 2 == 0, 1, -1)  # Convert to binary classification problem

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train AdaBoost classifier
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1.0)
    ada.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = ada.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy of AdaBoost Classifier: {accuracy:.2f}")
