from sklearn.datasets import load_digits
from DecisionTree import DecisionTreeClassifier as DecisionTree
from utils.cross_validation import cross_val_score
import numpy as np

# Load digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Initialize your DecisionTree model
model = DecisionTree()

custom_cv_scores = cross_val_score(model, X, y, cv=5, random_state=42)


# Print the custom cross-validation results
print("Custom Cross-Validation Scores:", custom_cv_scores)

mean_score = np.mean(custom_cv_scores)
std_score = np.std(custom_cv_scores)


# Print the benchmark results
print(f"Mean Cross-Validation Score: {mean_score}")
print(f"Standard Deviation of Scores: {std_score}")
