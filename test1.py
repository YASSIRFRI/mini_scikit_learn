import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from NN import Network , Layer
 # Convert sparse matrix to dense array immediately
# Load Iris data
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# One-hot encode labels
encoder = OneHotEncoder()  # Initialize without the 'sparse' keyword
y_encoded = encoder.fit_transform(y).toarray() 

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)

# Define your neural network structure
network = Network()
network.add(Layer(4, 5, 'sigmoid'))  # Hidden layer with 5 neurons
network.add(Layer(5,10 , 'sigmoid'))
network.add(Layer(10, 3, 'sigmoid'))  # Output layer with 3 neurons (using sigmoid temporarily, switch to softmax later)


# Train the network
network.fit(X_train, y_train, epochs=1000, learning_rate=0.01)

# Predictions
predictions = network.predict(X_test)
print(predictions)

# Function to convert probabilities to class labels
def convert_prob_to_class(probs):
    class_indices = np.argmax(probs, axis=1)
    return class_indices

# Convert predictions to class labels
predicted_classes = convert_prob_to_class(predictions)

# Convert one-hot encoded test labels back to class labels
actual_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == actual_classes)
print("Accuracy:", accuracy)

print(actual_classes)
print(predicted_classes)

print(X_train.shape)

