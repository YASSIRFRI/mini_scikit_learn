import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation_function):
        self.input = None
        self.output = None
        self.weights = np.random.rand(input_size, output_size)
        self.activation_function = activation_function

    def forward_propagation(self, given_input):
        self.input = given_input
        net_input = np.dot(self.input, self.weights)
        if self.activation_function == "sigmoid":
            self.output = 1 / (1 + np.exp(-net_input))
        elif self.activation_function == "tanh":
            self.output = np.tanh(net_input)
        return self.output

    def back_propagation(self, previous_errors, learning_rate, targets):
        delta = (targets - self.output) * self.output * (1 - self.output) if previous_errors is None else previous_errors * self.output * (1 - self.output)
        upcoming_error = np.dot(delta, self.weights.T)
        self.weights += learning_rate * np.outer(self.input, delta)
        return upcoming_error

class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        predictions = []
        for input_sample in input_data:
            output = input_sample
            for layer in self.layers:
                output = layer.forward_propagation(output)
            predictions.append(output)
        return np.array(predictions)

    def fit(self, train_data, train_labels, epochs, learning_rate):
        for epoch in range(epochs):
            for input_sample, target in zip(train_data, train_labels):
                output = input_sample
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                previous_errors = None
                for layer in reversed(self.layers):
                    previous_errors = layer.back_propagation(previous_errors, learning_rate, target)

    def score(self, test_data, test_labels):
        predictions = self.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(test_labels, axis=1)
        return np.mean(predicted_classes == actual_classes)

