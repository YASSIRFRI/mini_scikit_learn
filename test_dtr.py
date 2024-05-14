import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor as SklearnDecisionTreeRegressor
from DecisionTreeRegressor import DecisionTreeRegressor


# Load the California housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train your custom DecisionTreeRegressor
my_tree = DecisionTreeRegressor(max_depth=10)
my_tree.fit(X_train, y_train)
y_pred_my_tree = my_tree.predict(X_test)

# Train sklearn's DecisionTreeRegressor
sklearn_tree = SklearnDecisionTreeRegressor(max_depth=10, random_state=42)
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn_tree = sklearn_tree.predict(X_test)

# Calculate MSE for both models
mse_my_tree = mean_squared_error(y_test, y_pred_my_tree)
mse_sklearn_tree = mean_squared_error(y_test, y_pred_sklearn_tree)

# Print the results
print(f"Custom DecisionTreeRegressor MSE: {mse_my_tree}")
print(f"Sklearn DecisionTreeRegressor MSE: {mse_sklearn_tree}")
