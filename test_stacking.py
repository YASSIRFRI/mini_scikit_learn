from LinearModel import LogisticRegression
from NBClassifier import NBClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from RandomForest import RandomForestClassifier
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ModelEnsembler import StackingEnsembler
from KnnEstimator import KnnEstimator
import numpy as np

# Load the datasets
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split the data for digits dataset
X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

# Split the data for iris dataset
X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# Define base models
base_models = [
    KnnEstimator(n_neighbors=5),
    NBClassifier(),
    RandomForestClassifier(n_estimators=100, max_depth=10)
]

# Define meta model
meta_model = RandomForestClassifier(n_estimators=100, max_depth=10)

# Train and test StackingEnsembler for digits dataset
stacking_ensembler_digits = StackingEnsembler(base_models, meta_model)
stacking_ensembler_digits.fit(X_digits_train, y_digits_train)
stacking_score_digits = accuracy_score(y_digits_test, stacking_ensembler_digits.predict(X_digits_test))
print("StackingEnsembler Score (Digits):", stacking_score_digits)

# Train and test StackingEnsembler for iris dataset
stacking_ensembler_iris = StackingEnsembler(base_models, meta_model)
stacking_ensembler_iris.fit(X_iris_train, y_iris_train)
stacking_score_iris = accuracy_score(y_iris_test, stacking_ensembler_iris.predict(X_iris_test))
print("StackingEnsembler Score (Iris):", stacking_score_iris)
