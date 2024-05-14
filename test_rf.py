from sklearn.datasets import load_digits, load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.metrics import mean_squared_error
from RandomForest import RandomForestClassifier, RandomForestRegressor

# Load the datasets
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

california = fetch_california_housing()
X_california, y_california = california.data, california.target

# Split the data for digits dataset
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

# Split the data for iris dataset
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Split the data for California housing dataset
X_california_train, X_california_test, y_california_train, y_california_test = train_test_split(X_california, y_california, test_size=0.2, random_state=42)

# Train and test your RandomForestClassifier implementation for digits dataset
your_model_digits = RandomForestClassifier(n_estimators=100, max_depth=10)  # Adjust parameters as needed
your_model_digits.fit(X_digits_train, y_digits_train)
your_score_digits = your_model_digits.score(X_digits_test, y_digits_test)
print("Your RandomForestClassifier Score (Digits):", your_score_digits)

# Train and test your RandomForestClassifier implementation for iris dataset
your_model_iris = RandomForestClassifier(n_estimators=100, max_depth=10)  # Adjust parameters as needed
your_model_iris.fit(X_iris_train, y_iris_train)
your_score_iris = your_model_iris.score(X_iris_test, y_iris_test)
print("Your RandomForestClassifier Score (Iris):", your_score_iris)

# Train and test scikit-learn's RandomForestClassifier model for digits dataset
sklearn_model_digits = SklearnRandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Adjust parameters as needed
sklearn_model_digits.fit(X_digits_train, y_digits_train)
sklearn_score_digits = sklearn_model_digits.score(X_digits_test, y_digits_test)
print("Scikit-learn RandomForestClassifier Score (Digits):", sklearn_score_digits)

# Train and test scikit-learn's RandomForestClassifier model for iris dataset
sklearn_model_iris = SklearnRandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)  # Adjust parameters as needed
sklearn_model_iris.fit(X_iris_train, y_iris_train)
sklearn_score_iris = sklearn_model_iris.score(X_iris_test, y_iris_test)
print("Scikit-learn RandomForestClassifier Score (Iris):", sklearn_score_iris)

# Train and test your RandomForestRegressor implementation for California housing dataset
#your_model_california = RandomForestRegressor(n_estimators=100, max_depth=10)  # Adjust parameters as needed
#your_model_california.fit(X_california_train, y_california_train)
#your_predictions_california = your_model_california.predict(X_california_test)
#your_mse_california = mean_squared_error(y_california_test, your_predictions_california)
#print("Your RandomForestRegressor MSE (California):", your_mse_california)

# Train and test scikit-learn's RandomForestRegressor model for California housing dataset
sklearn_model_california = SklearnRandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)  # Adjust parameters as needed
sklearn_model_california.fit(X_california_train, y_california_train)
sklearn_predictions_california = sklearn_model_california.predict(X_california_test)
sklearn_mse_california = mean_squared_error(y_california_test, sklearn_predictions_california)
print("Scikit-learn RandomForestRegressor MSE (California):", sklearn_mse_california)
