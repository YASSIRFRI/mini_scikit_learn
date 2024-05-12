from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from NN import Network, Layer 
import numpy as np 
# Load Digits data
digits = load_digits()
X = digits.data
y = digits.target.reshape(-1, 1)

# Select a subset of data for quicker training
# For example, use only 30% of the data
subset_fraction = 0.3
subset_size = int(X.shape[0] * subset_fraction)
indices = np.random.choice(X.shape[0], subset_size, replace=False)
X_subset =X
y_subset =y

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# One-hot encode labels
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y_subset).toarray()
print(y_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Define your neural network structure
network = Network()
network.add(Layer(64, 64, 'sigmoid'))  # Hidden layer with 50 neurons
network.add(Layer(64, 10, 'sigmoid'))  # Output layer with 10 neurons

# Train the network
network.fit(X_train, y_train, epochs=1000, learning_rate=0.01)

# Predictions
predictions = network.predict(X_test)

# Convert predictions to class labels
def convert_prob_to_class(probs):
    return np.argmax(probs, axis=1)

predicted_classes = convert_prob_to_class(predictions)

# Convert one-hot encoded test labels back to class labels
actual_classes = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = np.mean(predicted_classes == actual_classes)
print("Accuracy:", accuracy)
print("Predicted classes:", predicted_classes)
print("Actual classes:", actual_classes)




