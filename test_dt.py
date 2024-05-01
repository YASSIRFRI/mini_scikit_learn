from DecisionTreeClassifier import DecisionTreeClassifier 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

# Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Train and test your implementation
your_model = DecisionTreeClassifier()
your_model.fit(X_train, y_train)
your_score = your_model.score(X_test, y_test)
print("Your DecisionTreeClassifier Score:", your_score)

# Train and test scikit-learn's DecisionTreeClassifier model
sklearn_model = SklearnDecisionTreeClassifier()
sklearn_model.fit(X_train, y_train)
sklearn_score = sklearn_model.score(X_test, y_test)
print("Scikit-learn DecisionTreeClassifier Score:", sklearn_score)
