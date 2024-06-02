import random
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Any, Union

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.25, random_state: int = None, stratify: Union[List[Any], np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data into training and testing sets.

    Parameters:
    - X: array-like, feature set.
    - y: array-like, labels.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, seed used by the random number generator.
    - stratify: array-like, if not None, data is split in a stratified fashion using this as the class labels.

    Returns:
    - X_train: np.ndarray, the training set features.
    - X_test: np.ndarray, the testing set features.
    - y_train: np.ndarray, the training set labels.
    - y_test: np.ndarray, the testing set labels.
    """
    
    if not isinstance(X, (np.ndarray, list)) or not isinstance(y, (np.ndarray, list)):
        raise ValueError("X and y should be numpy arrays or lists.")
    
    X, y = np.array(X), np.array(y)
    
    if len(X) != len(y):
        raise ValueError("The length of X and y must be equal.")
    
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)
    
    if stratify is None:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        
        n_test = int(len(y) * test_size)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
    else:
        if len(stratify) != len(y):
            raise ValueError("stratify should be of the same length as y.")
        
        stratified_data = defaultdict(list)
        for idx, label in enumerate(stratify):
            stratified_data[label].append(idx)
        
        train_indices = []
        test_indices = []
        
        for label, indices in stratified_data.items():
            n_test_label = int(len(indices) * test_size)
            random.shuffle(indices)
            
            test_indices.extend(indices[:n_test_label])
            train_indices.extend(indices[n_test_label:])
        
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris

    def test_decision_tree_classifier():
        # Load the iris dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        #clf = DecisionTreeClassifier()
        #clf.fit(X_train, y_train)
        #accuracy = clf.score(X_test, y_test)
        #print(f"Accuracy: {accuracy}")
    
    test_decision_tree_classifier()
