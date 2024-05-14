from LinearModel import LogisticRegression as CustomLogisticRegression
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

# Load the datasets
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split the data for digits dataset
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

# Split the data for iris dataset
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Train and test your implementation for digits dataset
your_model_digits = CustomLogisticRegression(max_iter=10000) 
your_model_digits.fit(X_digits_train, y_digits_train)
your_score_digits = your_model_digits.score(X_digits_test, y_digits_test)  # Assuming your model has a score method
print("Your LogisticRegression Score (Digits):", your_score_digits)

# Train and test your implementation for iris dataset
your_model_iris = CustomLogisticRegression(max_iter=10000)  
your_model_iris.fit(X_iris_train, y_iris_train)
your_score_iris = your_model_iris.score(X_iris_test, y_iris_test)  # Assuming your model has a score method
print("Your LogisticRegression Score (Iris):", your_score_iris)

# Train and test scikit-learn's LogisticRegression model for digits dataset
sklearn_model_digits = SklearnLogisticRegression(max_iter=10000)  # Use max_iter=10000 to ensure convergence
sklearn_model_digits.fit(X_digits_train, y_digits_train)
sklearn_score_digits = sklearn_model_digits.score(X_digits_test, y_digits_test)
print("Scikit-learn LogisticRegression Score (Digits):", sklearn_score_digits)

# Train and test scikit-learn's LogisticRegression model for iris dataset
sklearn_model_iris = SklearnLogisticRegression(max_iter=10000)  # Use max_iter=10000 to ensure convergence
sklearn_model_iris.fit(X_iris_train, y_iris_train)
sklearn_score_iris = sklearn_model_iris.score(X_iris_test, y_iris_test)
print("Scikit-learn LogisticRegression Score (Iris):", sklearn_score_iris)
