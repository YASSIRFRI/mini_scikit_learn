import numpy as np
import random

def cross_val_score(model, X, y, cv=5, random_state=None):
    """
    Perform k-fold cross-validation.

    Parameters:
    - model: object, the model to evaluate. Must have fit and score methods.
    - X: array-like, feature set.
    - y: array-like, labels.
    - cv: int, number of folds.
    - random_state: int, seed used by the random number generator.

    Returns:
    - scores: list, cross-validation scores.
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    X, y = np.asarray(X), np.asarray(y)
    
    if len(X) != len(y):
        raise ValueError("The length of X and y must be equal.")
    
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[:n_samples % cv] += 1
    current = 0
    scores = []

    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        
        current = stop

    return scores

# Example usage:
#class SimpleModel:
    #def fit(self, X, y):
        ## A dummy fit method
        #pass
    
    #def score(self, X, y):
        ## A dummy score method returning a random score
        #return random.random()

## Load dataset
#digits = load_digits()
#X, y = digits.data, digits.target

## Create and evaluate model
#model = SimpleModel()
#scores = cross_val_score(model, X, y, cv=5, random_state=42)
#print("Cross-validation scores:", scores)
