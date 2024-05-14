from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearModel import LogisticRegression

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an instance of the LogisticRegression class
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train_scaled.T, y_train.reshape(1, -1), num_iterations=1000, learning_rate=0.5, print_cost=True)

# Evaluate the model
train_accuracy, test_accuracy = model.evaluate(X_train_scaled.T, y_train.reshape(1, -1), X_test_scaled.T, y_test.reshape(1, -1))
