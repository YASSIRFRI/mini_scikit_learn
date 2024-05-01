import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from LinearModel import LinearRegression  

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and test your implementation
your_model = LinearRegression()
your_model.fit(X_train, y_train)
your_score = your_model.score(X_test, y_test)
print("Your Model Score:", your_score)

# Train and test scikit-learn's LinearRegression model
sklearn_model = SklearnLinearRegression()
sklearn_model.fit(X_train, y_train)
sklearn_score = sklearn_model.score(X_test, y_test)
print("Scikit-learn Model Score:", sklearn_score)
