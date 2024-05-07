from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from NBClassifier import NBClassifier

# Load the datasets
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

wine = load_wine()
X_wine, y_wine = wine.data, wine.target

# Split the datasets into training and testing sets
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)

# Train and test your implementation for digits dataset
your_model_digits = NBClassifier()
your_model_digits.fit(X_digits_train, y_digits_train)
your_predictions_digits = your_model_digits.predict(X_digits_test)
your_accuracy_digits = accuracy_score(y_digits_test, your_predictions_digits)
print("Your NBClassifier Accuracy (Digits):", your_accuracy_digits)

# Train and test your implementation for iris dataset
your_model_iris = NBClassifier()
your_model_iris.fit(X_iris_train, y_iris_train)
your_predictions_iris = your_model_iris.predict(X_iris_test)
your_accuracy_iris = accuracy_score(y_iris_test, your_predictions_iris)
print("Your NBClassifier Accuracy (Iris):", your_accuracy_iris)

# Train and test your implementation for wine dataset
your_model_wine = NBClassifier()
your_model_wine.fit(X_wine_train, y_wine_train)
your_predictions_wine = your_model_wine.predict(X_wine_test)
your_accuracy_wine = accuracy_score(y_wine_test, your_predictions_wine)
print("Your NBClassifier Accuracy (Wine):", your_accuracy_wine)

# Train and test scikit-learn's GaussianNB model for digits dataset
sklearn_model_digits = GaussianNB()
sklearn_model_digits.fit(X_digits_train, y_digits_train)
sklearn_predictions_digits = sklearn_model_digits.predict(X_digits_test)
sklearn_accuracy_digits = accuracy_score(y_digits_test, sklearn_predictions_digits)
print("Scikit-learn GaussianNB Accuracy (Digits):", sklearn_accuracy_digits)

# Train and test scikit-learn's GaussianNB model for iris dataset
sklearn_model_iris = GaussianNB()
sklearn_model_iris.fit(X_iris_train, y_iris_train)
sklearn_predictions_iris = sklearn_model_iris.predict(X_iris_test)
sklearn_accuracy_iris = accuracy_score(y_iris_test, sklearn_predictions_iris)
print("Scikit-learn GaussianNB Accuracy (Iris):", sklearn_accuracy_iris)

# Train and test scikit-learn's GaussianNB model for wine dataset
sklearn_model_wine = GaussianNB()
sklearn_model_wine.fit(X_wine_train, y_wine_train)
sklearn_predictions_wine = sklearn_model_wine.predict(X_wine_test)
sklearn_accuracy_wine = accuracy_score(y_wine_test, sklearn_predictions_wine)
print("Scikit-learn GaussianNB Accuracy (Wine):", sklearn_accuracy_wine)
