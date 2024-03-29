import numpy as np

from src.Networks.Layer.Layer import Layer
from src.Regularizations import NormRegularizationFunction


class FlattenLayer(Layer):

    def __init__(self):
        super().__init__()
        self.last_input_shape = None

    def forward(self, x_input: np.array) -> np.array:
        self.last_input_shape = x_input.shape
        output = []
        for i in x_input:
            output.append(i.flatten())
        return np.array(output)

    def backward(self, gradient: np.array, learning_rate: float,
                 regularization_function: NormRegularizationFunction) -> np.array:
        return gradient.reshape(self.last_input_shape[:-1])

    def update_weight(self, learning_rate: float, grads: np.array):
        pass

    def update_bias(self, learning_rate: float, grads: np.array):
        pass
