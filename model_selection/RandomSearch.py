import numpy as np
import random
from typing import Dict, Any, List, Union

class RandomSearch:
    def __init__(self, model, param_distributions: Dict[str, List[Any]], n_iter: int = 10, random_state: Union[int, None] = None):
        """
        Initializes the RandomSearch with a model, parameter distributions, and the number of iterations.

        Parameters:
        - model: The machine learning model to be tuned.
        - param_distributions: Dictionary where keys are parameter names and values are lists of parameter settings to sample from.
        - n_iter: Number of parameter settings that are sampled.
        - random_state: Seed for the random number generator.
        """
        self.model = model
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.results_ = []

        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def fit(self, X, y):
        """
        Performs random search over the parameter distributions.

        Parameters:
        - X: Training data.
        - y: Training labels.
        """
        param_names = list(self.param_distributions.keys())
        param_values = list(self.param_distributions.values())
        
        for _ in range(self.n_iter):
            params = {name: random.choice(values) for name, values in zip(param_names, param_values)}
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
        Returns the best parameters found during the random search.
        """
        return self.best_params_

    def get_best_score(self) -> float:
        """
        Returns the best score obtained during the random search.
        """
        return self.best_score_

    def get_results(self) -> List[Dict[str, Any]]:
        """
        Returns the results of the random search.
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

    # Define the parameter distributions
    param_distributions = {
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 3, 4, 5]
    }

    # Perform random search
    random_search = RandomSearch(model, param_distributions, n_iter=10, random_state=42)
    random_search.fit(X, y)

    # Print the best parameters and best score
    print("Best Parameters:", random_search.get_best_params())
    print("Best Score:", random_search.get_best_score())

    # Print all results
    print("Random Search Results:", random_search.get_results())
