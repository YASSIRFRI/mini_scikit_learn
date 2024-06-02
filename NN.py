import numpy as np

class Layer:
    """
    A layer in a neural network.
    Parameters:
    -----------
    input_size : int
        The number of input units in the layer.
    output_size : int
        The number of output units in the layer.
    activation_function : str
        The activation function to be used in the layer ('sigmoid' or 'tanh').

    Attributes:
    -----------
    input : numpy.ndarray or None
        The input to the layer.
    output : numpy.ndarray or None
        The output from the layer after applying the activation function.
    weights : numpy.ndarray
        The weights matrix of the layer.
    activation_function : str
        The activation function to be used in the layer.
    """

    def __init__(self, input_size, output_size, activation_function):
        self.input = None
        self.output = None
        self.weights = np.random.rand(input_size, output_size)
        self.activation_function = activation_function

    def forward_propagation(self, given_input):
        """
        Perform forward propagation through the layer.
        Parameters:
        -----------
        given_input : numpy.ndarray
            The input data to the layer.

        Returns:
        --------
        numpy.ndarray
            The output of the layer after applying the activation function.
        """
        self.input = given_input
        net_input = np.dot(self.input, self.weights)
        if self.activation_function == "sigmoid":
            self.output = 1 / (1 + np.exp(-net_input))
        elif self.activation_function == "tanh":
            self.output = np.tanh(net_input)
        return self.output

    def back_propagation(self, previous_errors, learning_rate, targets):
        """
        Perform backpropagation through the layer.
        Parameters:
        -----------
        previous_errors : numpy.ndarray or None
            The errors from the previous layer.
        learning_rate : float
            The learning rate for weight updates.
        targets : numpy.ndarray
            The target values.

        Returns:
        --------
        numpy.ndarray
            The upcoming error to be propagated to the previous layer.
        """
        delta = (targets - self.output) * self.output * (1 - self.output) if previous_errors is None else previous_errors * self.output * (1 - self.output)
        upcoming_error = np.dot(delta, self.weights.T)
        self.weights += learning_rate * np.outer(self.input, delta)
        return upcoming_error

class Network:
    """
    A feedforward neural network.
    Attributes:
    -----------
    layers : list
        A list to store the layers of the neural network.
    """

    def __init__(self):
        self.layers = []

    def add(self, layer):
        """
        Add a layer to the neural network.
        Parameters:
        -----------
        layer : Layer
            The layer to be added to the network.
        """
        self.layers.append(layer)

    def predict(self, input_data):
        """
        Predict the output for given input data.
        Parameters:
        -----------
        input_data : numpy.ndarray
            The input data.

        Returns:
        --------
        numpy.ndarray
            The predicted output.
        """
        predictions = []
        for input_sample in input_data:
            output = input_sample
            for layer in self.layers:
                output = layer.forward_propagation(output)
            predictions.append(output)
        return np.array(predictions)

    def fit(self, train_data, train_labels, epochs, learning_rate):
        """
        Train the neural network with given training data.
        Parameters:
        -----------
        train_data : numpy.ndarray
            The training data.
        train_labels : numpy.ndarray
            The target labels.
        epochs : int
            The number of epochs to train the model.
        learning_rate : float
            The learning rate for training.
        """
        for epoch in range(epochs):
            for input_sample, target in zip(train_data, train_labels):
                output = input_sample
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                previous_errors = None
                for layer in reversed(self.layers):
                    previous_errors = layer.back_propagation(previous_errors, learning_rate, target)

    def score(self, test_data, test_labels):
        """
        Evaluate the accuracy of the model on test data.
        Parameters:
        -----------
        test_data : numpy.ndarray
            The test data.
        test_labels : numpy.ndarray
            The true labels for the test data.

        Returns:
        --------
        float
            The accuracy of the model.
        """
        predictions = self.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        actual_classes = np.argmax(test_labels, axis=1)
        return np.mean(predicted_classes == actual_classes)
