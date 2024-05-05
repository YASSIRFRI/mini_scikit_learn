from DecisionTreeClassifier import DecisionTreeClassifier 
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

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
your_model_digits = DecisionTreeClassifier()
your_model_digits.fit(X_digits_train, y_digits_train)
your_score_digits = your_model_digits.score(X_digits_test, y_digits_test)
print("Your DecisionTreeClassifier Score (Digits):", your_score_digits)

# Train and test your implementation for iris dataset
your_model_iris = DecisionTreeClassifier()
your_model_iris.fit(X_iris_train, y_iris_train)
your_score_iris = your_model_iris.score(X_iris_test, y_iris_test)
print("Your DecisionTreeClassifier Score (Iris):", your_score_iris)

# Train and test scikit-learn's DecisionTreeClassifier model for digits dataset
sklearn_model_digits = SklearnDecisionTreeClassifier()
sklearn_model_digits.fit(X_digits_train, y_digits_train)
sklearn_score_digits = sklearn_model_digits.score(X_digits_test, y_digits_test)
print("Scikit-learn DecisionTreeClassifier Score (Digits):", sklearn_score_digits)

# Train and test scikit-learn's DecisionTreeClassifier model for iris dataset
sklearn_model_iris = SklearnDecisionTreeClassifier()
sklearn_model_iris.fit(X_iris_train, y_iris_train)
sklearn_score_iris = sklearn_model_iris.score(X_iris_test, y_iris_test)
print("Scikit-learn DecisionTreeClassifier Score (Iris):", sklearn_score_iris)
