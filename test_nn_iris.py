import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from NN import Network, Layer

# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode labels
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define your neural network structure
network = Network()
network.add(Layer(4, 5, 'sigmoid'))  # Hidden layer with 5 neurons
network.add(Layer(5, 10, 'sigmoid'))
network.add(Layer(10, 3, 'sigmoid'))  # Output layer with 3 neurons

# Train the custom network
network.fit(X_train, y_train, epochs=1000, learning_rate=0.01)

# Predictions with the custom network
predictions = network.predict(X_test)

# Function to convert probabilities to class labels
def convert_prob_to_class(probs):
    class_indices = np.argmax(probs, axis=1)
    return class_indices

# Convert predictions to class labels
predicted_classes = convert_prob_to_class(predictions)

# Convert one-hot encoded test labels back to class labels
actual_classes = np.argmax(y_test, axis=1)

# Calculate accuracy for the custom network
accuracy = np.mean(predicted_classes == actual_classes)
print("Accuracy of custom neural network:", accuracy)

# Train and test scikit-learn's MLPClassifier
# Convert y back to 1D array for scikit-learn compatibility
y_train_labels = np.argmax(y_train, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Define and train the scikit-learn MLPClassifier with increased iterations
mlp = MLPClassifier(hidden_layer_sizes=(5, 10), activation='logistic', max_iter=2000, learning_rate_init=0.01, random_state=42)
mlp.fit(X_train, y_train_labels)

# Predictions with scikit-learn MLPClassifier
y_pred = mlp.predict(X_test)

# Calculate accuracy for scikit-learn MLPClassifier
accuracy_sklearn = accuracy_score(y_test_labels, y_pred)
print("Accuracy of scikit-learn's NN model:", accuracy_sklearn)
