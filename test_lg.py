import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from LogisticRegression import LogisticRegression

# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler= MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Create an instance of the LogisticRegression class for each class
models = [LogisticRegression() for _ in range(10)]

# Train each model using One-vs-Rest strategy
for i in range(10):
    y_train_binary = (y_train == i).astype(int)
    models[i].fit(X_train, y_train_binary)

# Predict class labels for each class using each model
predictions = np.array([model.predict(X_test) for model in models])

# Combine predictions to make multi-class predictions
multi_class_predictions = np.argmax(predictions, axis=0)

# Evaluate the model
accuracy = np.mean(multi_class_predictions == y_test)
print("Accuracy:", accuracy)
