import numpy as np
from sklearn.datasets import fetch_openml

# Load MNIST data from scikit-learn
mnist = fetch_openml('mnist_784')

# Split the data into features (images) and labels
x, y = mnist.data, mnist.target

# Convert labels to integers
y = y.astype(np.uint8)

# Normalize pixel values to range [0, 1]
x = x / 255.0

# Split the data into training and testing sets
# You can adjust the test_size parameter as needed
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(categories='auto', sparse=False)
y_train_one_hot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_one_hot = encoder.transform(y_test.reshape(-1, 1))

# Define your neural network model
class MNIST_NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        predicted = []
        for i in range(len(input_data)):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            predicted.append(output)
        return predicted

    def fit(self, train_data, train_labels, epochs, learning_rate):
        samples = len(train_data)
        for epoch in range(epochs):
            for sample in range(samples):
                cur_deltas = []
                output = train_data[sample]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                for layer in reversed(self.layers):
                    cur_deltas = layer.back_propagation(cur_deltas, learning_rate, train_labels[sample])

# Create an instance of your neural network
mnist_nn = MNIST_NeuralNetwork()
mnist_nn.add(Layer(784, 128, "sigmoid")) 
mnist_nn.add(Layer(128, 64, "sigmoid"))
mnist_nn.add(Layer(64, 10, "sigmoid"))   

# Train the model
mnist_nn.fit(x_train, y_train_one_hot, epochs=5, learning_rate=0.1)

# Test the model
predicted = mnist_nn.predict(x_test)
