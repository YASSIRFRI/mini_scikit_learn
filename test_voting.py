from LinearModel import LogisticRegression
from NBClassifier import NBClassifier
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier
from RandomForest import RandomForestClassifier
from sklearn.datasets import load_digits, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ModelEnsembler import VotingEnsembler
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

# Define models for VotingEnsembler
models = [
    KnnEstimator(n_neighbors=5),
    NBClassifier(),
    LogisticRegression(max_iter=10000),
]

voting_ensembler_digits = VotingEnsembler(models)
voting_ensembler_digits.fit(X_digits_train, y_digits_train)
voting_score_digits = voting_ensembler_digits.score(X_digits_test, y_digits_test)
print("VotingEnsembler Score (Digits):", voting_score_digits)

# Train and test VotingEnsembler for iris dataset
voting_ensembler_iris = VotingEnsembler(models)
voting_ensembler_iris.fit(X_iris_train, y_iris_train)
voting_score_iris=voting_ensembler_iris.score(X_iris_test, y_iris_test)
print("VotingEnsembler Score (Iris):", voting_score_iris)

# Train and test scikit-learn's RandomForestClassifier model for digits dataset
sklearn_model_digits = SklearnRandomForestClassifier()
sklearn_model_digits.fit(X_digits_train, y_digits_train)
sklearn_score_digits = sklearn_model_digits.score(X_digits_test, y_digits_test)
print("Scikit-learn RandomForestClassifier Score (Digits):", sklearn_score_digits)

# Train and test scikit-learn's RandomForestClassifier model for iris dataset
sklearn_model_iris = SklearnRandomForestClassifier()
sklearn_model_iris.fit(X_iris_train, y_iris_train)
sklearn_score_iris = sklearn_model_iris.score(X_iris_test, y_iris_test)
print("Scikit-learn RandomForestClassifier Score (Iris):", sklearn_score_iris)
