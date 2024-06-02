import numpy as np
import itertools
from typing import Dict, Any, List

class GridSearch:
    def __init__(self, model, param_grid: Dict[str, List[Any]]):
        """
        Initializes the GridSearch with a model and a parameter grid.

        Parameters:
        - model: The machine learning model to be tuned.
        - param_grid: Dictionary where keys are parameter names and values are lists of parameter settings to try.
        """
        self.model = model
        self.param_grid = param_grid
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []

    def fit(self, X, y):
        """
        Performs grid search over the parameter grid.

        Parameters:
        - X: Training data.
        - y: Training labels.
        """
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            self.model.set_params(**params)
            self.model.fit(X, y)
            score = self.model.score(X, y)
            
            self.results_.append({
                'params': params,
                'score': score
            })

            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = params

    def get_best_params(self) -> Dict[str, Any]:
        """
        Returns the best parameters found during the grid search.
        """
        return self.best_params_

    def get_best_score(self) -> float:
        """
        Returns the best score obtained during the grid search.
        """
        return self.best_score_

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Returns the results of the grid search.
        """
        return self.results_

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Define the model
    model = DecisionTreeClassifier()

    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }

    # Perform grid search
    grid_search = GridSearch(model, param_grid)
    grid_search.fit(X, y)

    # Print the best parameters and best score
    print("Best Parameters:", grid_search.get_best_params())
    print("Best Score:", grid_search.get_best_score())

    # Print all results
    print("Grid Search Results:", grid_search.get_results())
