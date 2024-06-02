import numpy as np
from typing import Iterator, Tuple

class KFold:
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: int = None):
        """
        Initializes the KFold cross-validator.

        Parameters:
        - n_splits: int, number of folds.
        - shuffle: bool, whether to shuffle the data before splitting into batches.
        - random_state: int, seed used by the random number generator.
        """
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates indices to split data into training and test set.

        Parameters:
        - X: array-like, shape (n_samples, n_features) or (n_samples,)
        - y: array-like, shape (n_samples,), default=None

        Yields:
        - train_indices: array, the training set indices for that split.
        - test_indices: array, the testing set indices for that split.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate((indices[:start], indices[stop:]))
            yield train_indices, test_indices
            current = stop

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Define the model
    model = DecisionTreeClassifier()

    # Initialize KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform k-fold cross-validation
    scores = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

    print("K-Fold Cross-Validation Scores:", scores)
