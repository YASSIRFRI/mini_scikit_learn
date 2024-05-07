import numpy as np
from typing import Tuple
from layer import Layer
from activation import Activation
from loss import Loss
from optimizer import Optimizer
from callback import Callback
from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract base model class"""

    @property
    @abstractmethod
    def learning_rate(self):
        ...

    @learning_rate.setter
    @abstractmethod
    def learning_rate(self, value: float):
        ...

    @abstractmethod
    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def fit(self, examples: np.ndarray, labels: np.ndarray, epochs: int):
        ...

    @abstractmethod
    def predict(self, examples: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def evaluate(self, examples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward_step(self, labels: np.ndarray):
        ...

    @abstractmethod
    def update(self):
        ...
        
        

class NeuralNetwork(Model):
    def __init__(
        self,
        layers: Tuple[Tuple[Layer, Activation]],
        loss: Loss,
        optimizer: Optimizer,
        regularization_factor: float = 0.0,
    ):
        self._layers = layers
        self._num_layers = len(layers)
        self._loss = loss
        self._optimizer = optimizer
        self._regularization_factor = regularization_factor
        self._input = None
        self._output = None
        self._num_examples = None

    def __call__(self, input_tensor: np.ndarray) -> np.ndarray:
        if self._num_examples is None:
            self._num_examples = input_tensor.shape[-1]

        output = input_tensor

        for layer, activation in self._layers:
            output = layer(output)
            output = activation(output)

        self._output = output
        return self._output

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value


    def backward_step(self, labels: np.ndarray):
        da = self._loss.gradient(self._output, labels)

        for index in reversed(range(0, self._num_layers)):
            layer, activation = self._layers[index]

            if index == 0:
                prev_layer_output = self._input
            else:
                prev_layer, prev_activation = self._layers[index - 1]
                prev_layer_output = prev_activation(prev_layer.output)

            dz = np.multiply(da, activation.gradient(layer.output))
            layer.grad_weights = (
                np.dot(dz, np.transpose(prev_layer_output)) / self._num_examples
            )
            layer.grad_weights = (
                layer.grad_weights
                + (self._regularization_factor / self._num_examples) * layer.weights
            )
            layer.grad_bias = np.mean(dz, axis=1, keepdims=True)
            da = np.dot(np.transpose(layer.weights), dz)
            
            self._optimizer.layer_number = index
            self._optimizer.update_weights(layer, layer.grad_weights)
            self._optimizer.update_bias(layer, layer.grad_bias)


    def fit(
        self,
        examples: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        verbose: bool = False,
        callbacks: Tuple[Callback] = (),
    ):
        for epoch in range(1, epochs + 1):
            self._input = examples
            _ = self(self._input)
            print("fit")
            print(self._output.shape, labels.shape)
            loss = self._loss(self._output, labels)
            self.backward_step(labels)
            self.update()
            for callback in callbacks:
                loss_scalar = float(np.squeeze(loss))
                callback.on_epoch_end(epoch, loss_scalar)
            if verbose:
                print(f"Epoch: {epoch:03d}, Loss {loss:0.4f}")

    def predict(self, examples: np.ndarray) -> np.ndarray:
        outputs = self(examples)
        return outputs

    def evaluate(self, examples: np.ndarray, labels: np.ndarray) -> np.ndarray:
        _ = self(examples)
        print("evaluate")
        print(self._output.shape, labels.shape)
        return self._loss(self._output, labels)

    def update(self):
        for ln in range(0, len(self._layers)):
            self._optimizer.layer_number = ln
            self._layers[ln][0].update(self._optimizer)